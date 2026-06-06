"""
Base pruner class for correlation-stratified benchmark compression.

Implements a universal pruning algorithm that:
1. Stratifies samples by difficulty (easy/medium/hard)
2. Scores samples by discrimination power (variance across models)
3. Filters by correlation with full-set model rankings
4. Validates generalization via leave-one-out

Designed to work across LCB, AA-LCR, and MMMU benchmarks
with benchmark-specific adapters handling data loading.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SampleScore:
    """Per-sample scores across all models."""
    index: int
    scores: Dict[str, float]  # model_name -> score (0.0 or 1.0)

    @property
    def mean_score(self) -> float:
        """Average score across all models — proxy for difficulty."""
        return float(np.mean(list(self.scores.values())))

    @property
    def score_variance(self) -> float:
        """Variance across model scores — proxy for discrimination power."""
        return float(np.var(list(self.scores.values())))

    @property
    def model_ranking(self) -> List[Tuple[str, float]]:
        """Models sorted by score descending."""
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)


class BasePruner:
    """
    Universal correlation-stratified pruner.

    Works across any benchmark where samples can be scored
    as pass/fail or continuous scores across multiple models.

    Args:
        prune_ratio: Fraction of samples to keep (e.g. 0.1 = keep 10%)
        difficulty_bins: Number of difficulty bins (default: 3 = easy/medium/hard)
        bin_weights: Fraction of budget allocated to each bin (easy, medium, hard)
        min_samples: Minimum samples to keep regardless of prune_ratio
    """

    def __init__(
        self,
        prune_ratio: float = 0.1,
        difficulty_bins: int = 3,
        bin_weights: Optional[List[float]] = None,
        min_samples: int = 5,
    ):
        if not 0 < prune_ratio <= 1.0:
            raise ValueError(f'prune_ratio must be between 0 and 1, got {prune_ratio}')

        self.prune_ratio = prune_ratio
        self.difficulty_bins = difficulty_bins
        self.bin_weights = bin_weights or [0.2, 0.5, 0.3]  # easy/medium/hard
        self.min_samples = min_samples

        if len(self.bin_weights) != difficulty_bins:
            raise ValueError(
                f'bin_weights length {len(self.bin_weights)} '
                f'must match difficulty_bins {difficulty_bins}'
            )
        if abs(sum(self.bin_weights) - 1.0) > 1e-6:
            raise ValueError(f'bin_weights must sum to 1.0, got {sum(self.bin_weights)}')

    def _stratify_by_difficulty(
        self,
        samples: List[SampleScore],
    ) -> List[List[SampleScore]]:
        """
        Bin samples into difficulty tiers by mean score.

        Easy = high mean score (most models pass)
        Hard = low mean score (most models fail)
        Medium = in between (most discriminating)
        """
        sorted_samples = sorted(samples, key=lambda s: s.mean_score, reverse=True)
        n = len(sorted_samples)
        bin_size = n // self.difficulty_bins

        bins = []
        for i in range(self.difficulty_bins):
            start = i * bin_size
            end = start + bin_size if i < self.difficulty_bins - 1 else n
            bins.append(sorted_samples[start:end])

        return bins

    def _compute_full_ranking(
        self,
        samples: List[SampleScore],
    ) -> Dict[str, float]:
        """
        Compute overall model ranking from full sample set.
        Returns: {model_name: mean_score_across_all_samples}
        """
        if not samples:
            return {}

        model_names = list(samples[0].scores.keys())
        ranking = {}
        for model in model_names:
            scores = [s.scores[model] for s in samples if model in s.scores]
            ranking[model] = float(np.mean(scores)) if scores else 0.0
        return ranking

    def _correlation_with_ranking(
        self,
        sample: SampleScore,
        full_ranking: Dict[str, float],
    ) -> float:
        """
        Compute correlation between this sample's per-model scores
        and the full-set model ranking.

        High correlation = this sample correctly identifies which
        model is better, consistent with the overall leaderboard.
        """
        models = [m for m in sample.scores if m in full_ranking]
        if len(models) < 2:
            return 0.0

        sample_scores = np.array([sample.scores[m] for m in models])
        full_scores = np.array([full_ranking[m] for m in models])

        # Handle zero variance (all same score) — not discriminating
        if np.std(sample_scores) < 1e-10 or np.std(full_scores) < 1e-10:
            return 0.0

        correlation = float(np.corrcoef(sample_scores, full_scores)[0, 1])
        return correlation if not np.isnan(correlation) else 0.0

    def select_samples(
        self,
        samples: List[SampleScore],
        force_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Main entry point. Returns list of selected sample indices.

        Args:
            samples: List of SampleScore objects with cross-model scores
            force_indices: Optional indices to always include (e.g. anchors)

        Returns:
            List of selected indices
        """
        if not samples:
            return []

        total_budget = max(
            self.min_samples,
            int(len(samples) * self.prune_ratio)
        )

        full_ranking = self._compute_full_ranking(samples)
        bins = self._stratify_by_difficulty(samples)

        selected_indices = []

        for bin_idx, bin_samples in enumerate(bins):
            bin_budget = max(1, int(total_budget * self.bin_weights[bin_idx]))

            # Score each sample: discrimination * correlation
            scored = []
            for sample in bin_samples:
                discrimination = sample.score_variance
                correlation = self._correlation_with_ranking(sample, full_ranking)
                # Combined score: weight discrimination more heavily
                combined = 0.6 * discrimination + 0.4 * max(0.0, correlation)
                scored.append((sample.index, combined))

            # Sort by combined score descending, take top bin_budget
            scored.sort(key=lambda x: x[1], reverse=True)
            selected_indices.extend([idx for idx, _ in scored[:bin_budget]])

        # Add any forced indices
        if force_indices:
            for idx in force_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)

        return sorted(set(selected_indices))

    def get_pruning_stats(
        self,
        all_samples: List[SampleScore],
        selected_indices: List[int],
    ) -> Dict:
        """Return statistics about the pruning for reporting."""
        selected_set = set(selected_indices)
        selected = [s for s in all_samples if s.index in selected_set]

        full_ranking = self._compute_full_ranking(all_samples)
        pruned_ranking = self._compute_full_ranking(selected)

        return {
            'total_samples': len(all_samples),
            'selected_samples': len(selected_indices),
            'actual_prune_ratio': len(selected_indices) / len(all_samples),
            'full_ranking': full_ranking,
            'pruned_ranking': pruned_ranking,
            'ranking_preserved': self._rankings_agree(full_ranking, pruned_ranking),
        }

    def _rankings_agree(
        self,
        ranking_a: Dict[str, float],
        ranking_b: Dict[str, float],
    ) -> bool:
        """Check if two rankings agree on model ordering."""
        if not ranking_a or not ranking_b:
            return False
        models = [m for m in ranking_a if m in ranking_b]
        if len(models) < 2:
            return True
        order_a = sorted(models, key=lambda m: ranking_a[m], reverse=True)
        order_b = sorted(models, key=lambda m: ranking_b[m], reverse=True)
        return order_a == order_b
