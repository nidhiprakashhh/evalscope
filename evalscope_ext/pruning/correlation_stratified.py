"""
Correlation-stratified pruning strategy.

Extends BasePruner with:
- Cross-model consistency weighting for judge noise (AA-LCR)
- Configurable prune ratios per benchmark type
- Utility functions for loading eval data from jsonl files

NOTE: This pruner is for multi-model benchmarks only (LCB, AA-LCR).
Both require ≥2 models to compute discrimination (score variance) and
correlation with the full-set ranking. MMMU has only 1 model in the
shipped data and uses a separate MmmuPruner class that stratifies by
topic_difficulty and subfield metadata instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base_pruner import BasePruner, SampleScore


class CorrelationStratifiedPruner(BasePruner):
    """
    Correlation-stratified pruner with judge noise handling.

    For deterministic benchmarks (LCB): standard algorithm.
    For LLM-judge benchmarks (AA-LCR, MMMU): applies cross-model
    consistency weighting to down-weight potentially noisy samples.

    Args:
        prune_ratio: Fraction of samples to keep
        judge_noise_correction: If True, apply cross-model consistency
            weighting (use for LLM-judged benchmarks like AA-LCR)
        bin_weights: Budget allocation across easy/medium/hard bins
    """

    def __init__(
        self,
        prune_ratio: float = 0.1,
        judge_noise_correction: bool = False,
        bin_weights: Optional[List[float]] = None,
        min_samples: int = 5,
    ):
        super().__init__(
            prune_ratio=prune_ratio,
            difficulty_bins=3,
            bin_weights=bin_weights,
            min_samples=min_samples,
        )
        self.judge_noise_correction = judge_noise_correction

    def _compute_consistency_weight(
        self,
        sample: SampleScore,
        full_ranking: Dict[str, float],
    ) -> float:
        """
        Compute reliability weight based on cross-model consistency.

        A sample is considered reliable when its per-model scores are
        consistent with the overall model ranking — better models should
        score higher on this sample too.

        Samples where the score pattern contradicts the overall ranking
        may reflect judge noise rather than genuine capability differences.

        Returns weight in [0.5, 1.0]: 1.0 = fully consistent, 0.5 = suspicious
        """
        models = [m for m in sample.scores if m in full_ranking]
        if len(models) < 2:
            return 1.0

        # Sort models by overall ranking
        ranked_models = sorted(models, key=lambda m: full_ranking[m], reverse=True)

        # Check if per-sample scores follow the same ordering
        consistent_pairs = 0
        total_pairs = 0

        for i in range(len(ranked_models)):
            for j in range(i + 1, len(ranked_models)):
                better_model = ranked_models[i]
                worse_model = ranked_models[j]
                total_pairs += 1

                better_score = sample.scores[better_model]
                worse_score = sample.scores[worse_model]

                # Consistent if better model scores >= worse model
                if better_score >= worse_score:
                    consistent_pairs += 1

        consistency_ratio = consistent_pairs / total_pairs if total_pairs > 0 else 1.0
        # Map to [0.5, 1.0] range
        return 0.5 + 0.5 * consistency_ratio

    def select_samples(
        self,
        samples: List[SampleScore],
        force_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Select samples using correlation-stratified algorithm.

        If judge_noise_correction is True, weights each sample's
        discrimination score by its cross-model consistency.
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

            scored = []
            for sample in bin_samples:
                discrimination = sample.score_variance
                correlation = self._correlation_with_ranking(sample, full_ranking)

                if self.judge_noise_correction:
                    consistency = self._compute_consistency_weight(
                        sample, full_ranking
                    )
                else:
                    consistency = 1.0

                # Combined score with consistency weighting
                combined = consistency * (
                    0.6 * discrimination + 0.4 * max(0.0, correlation)
                )
                scored.append((sample.index, combined))

            scored.sort(key=lambda x: x[1], reverse=True)
            selected_indices.extend([idx for idx, _ in scored[:bin_budget]])

        if force_indices:
            for idx in force_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)

        return sorted(set(selected_indices))


def load_scores_from_jsonl(
    predictions_dir: str,
    reviews_dir: str,
    benchmark_prefix: str,
    score_key: str = 'pass',
) -> List[SampleScore]:
    """
    Load and join predictions + reviews for a benchmark.

    Handles the file naming convention:
    {benchmark_prefix}__{model_name}.jsonl

    Args:
        predictions_dir: Path to predictions/ directory
        reviews_dir: Path to reviews/ directory
        benchmark_prefix: e.g. 'live_code_bench_v5' or 'aa_lcr'
        score_key: Key inside sample_score.score.value (e.g. 'pass' or 'acc')

    Returns:
        List of SampleScore objects with cross-model scores
    """
    reviews_path = Path(reviews_dir)

    # Find all review files matching this benchmark
    review_files = list(reviews_path.glob(f'{benchmark_prefix}__*.jsonl'))
    if not review_files:
        raise FileNotFoundError(
            f'No review files found for benchmark {benchmark_prefix} '
            f'in {reviews_dir}'
        )

    # Load scores per model: {model_name: {index: score}}
    model_scores: Dict[str, Dict[int, float]] = {}

    for review_file in review_files:
        # Extract model name from filename
        model_name = review_file.stem.replace(f'{benchmark_prefix}__', '')
        scores: Dict[int, float] = {}

        with open(review_file) as f:
            for line in f:
                row = json.loads(line.strip())
                idx = row['index']
                score_value = row['sample_score']['score']['value'].get(score_key)
                if score_value is not None:
                    scores[idx] = float(score_value)

        model_scores[model_name] = scores

    # Find common indices across all models
    if not model_scores:
        return []

    common_indices = set.intersection(*[
        set(scores.keys()) for scores in model_scores.values()
    ])

    # Build SampleScore objects
    sample_scores = []
    for idx in sorted(common_indices):
        scores_for_sample = {
            model: model_scores[model][idx]
            for model in model_scores
            if idx in model_scores[model]
        }
        sample_scores.append(SampleScore(index=idx, scores=scores_for_sample))

    return sample_scores


def prune_benchmark(
    predictions_dir: str,
    reviews_dir: str,
    benchmark_prefix: str,
    score_key: str = 'pass',
    prune_ratio: float = 0.1,
    judge_noise_correction: bool = False,
    bin_weights: Optional[List[float]] = None,
) -> Tuple[List[int], Dict]:
    """
    End-to-end pruning for a benchmark.

    Returns:
        Tuple of (selected_indices, pruning_stats)
    """
    samples = load_scores_from_jsonl(
        predictions_dir=predictions_dir,
        reviews_dir=reviews_dir,
        benchmark_prefix=benchmark_prefix,
        score_key=score_key,
    )

    pruner = CorrelationStratifiedPruner(
        prune_ratio=prune_ratio,
        judge_noise_correction=judge_noise_correction,
        bin_weights=bin_weights,
    )

    selected_indices = pruner.select_samples(samples)
    stats = pruner.get_pruning_stats(samples, selected_indices)

    return selected_indices, stats
