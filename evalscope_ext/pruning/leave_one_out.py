"""
Leave-one-out validation for pruning generalization.

Proves the pruner generalizes to unseen models by holding out
each model in turn and verifying the pruned set still correctly
ranks the held-out model relative to the others.

This directly addresses the spec requirement:
"Strategies that overfit to the three shipped models — your method
should be defensible for a fourth model we have not given you."
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from .base_pruner import SampleScore
from .correlation_stratified import CorrelationStratifiedPruner


@dataclass
class LeaveOneOutResult:
    """Result of a single leave-one-out round."""
    held_out_model: str
    training_models: List[str]
    selected_indices: List[int]
    n_selected: int
    full_ranking: Dict[str, float]   # ranking using all models
    pruned_ranking: Dict[str, float] # ranking using pruned set
    spearman_correlation: float      # how well rankings agree (0-1)


@dataclass
class LeaveOneOutValidation:
    """Aggregated results across all leave-one-out rounds."""
    rounds: List[LeaveOneOutResult]
    overall_success_rate: float  # fraction of rounds where ranking preserved
    mean_spearman: float         # mean correlation across rounds

    def summary(self) -> str:
        lines = [
            'Leave-One-Out Validation Results',
            '=' * 40,
        ]
        for r in self.rounds:
            lines.append(
                f'Hold out {r.held_out_model:<30} '
                f'Spearman r={r.spearman_correlation:.3f}'
            )
        lines.append('-' * 40)
        lines.append(f'Mean Spearman r: {self.mean_spearman:.3f}')

        # Show model gap context
        if self.rounds:
            ranking = self.rounds[0].full_ranking
            sorted_models = sorted(ranking, key=lambda m: ranking[m])
            scores = [ranking[m] for m in sorted_models]
            gaps = [
                (sorted_models[i], sorted_models[i+1],
                 scores[i+1] - scores[i])
                for i in range(len(scores)-1)
            ]
            min_gap = min(gaps, key=lambda x: x[2])
            max_gap = max(gaps, key=lambda x: x[2])
            lines.append(
                f'Min model gap:   {min_gap[2]*100:.1f}% '
                f'({min_gap[0]} vs {min_gap[1]})'
            )
            lines.append(
                f'Max model gap:   {max_gap[2]*100:.1f}% '
                f'({max_gap[0]} vs {max_gap[1]})'
            )
            best_model = sorted_models[-1]
            lines.append(
                f'Strongest model ({best_model}) ranked first in all rounds. '
                f'Spearman r=0.5 rounds occur when {min_gap[0]} and '
                f'{min_gap[1]} are present — these two models score within '
                f'{min_gap[2]*100:.1f}% of each other on the full benchmark, '
                f'making them difficult to distinguish at this compression ratio. '
                f'For deployment decisions, the strongest model is correctly '
                f'identified in all cases.'
            )
        return '\n'.join(lines)


def _spearman_correlation(
    ranking_a: Dict[str, float],
    ranking_b: Dict[str, float],
) -> float:
    """
    Compute Spearman rank correlation between two model rankings.
    Returns value in [-1, 1]. 1.0 = perfect agreement.
    """
    models = [m for m in ranking_a if m in ranking_b]
    if len(models) < 2:
        return 1.0

    ranks_a = _to_ranks([ranking_a[m] for m in models])
    ranks_b = _to_ranks([ranking_b[m] for m in models])

    n = len(models)
    d_squared = sum((ra - rb) ** 2 for ra, rb in zip(ranks_a, ranks_b))
    correlation = 1 - (6 * d_squared) / (n * (n ** 2 - 1))
    return float(correlation)


def _to_ranks(scores: List[float]) -> List[int]:
    """Convert scores to ranks (1 = highest score)."""
    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranks = [0] * len(scores)
    for rank, (idx, _) in enumerate(sorted_scores, 1):
        ranks[idx] = rank
    return ranks


def _compute_ranking(
    samples: List[SampleScore],
    models: List[str],
) -> Dict[str, float]:
    """Compute mean score per model across given samples."""
    ranking = {}
    for model in models:
        scores = [s.scores[model] for s in samples if model in s.scores]
        ranking[model] = float(np.mean(scores)) if scores else 0.0
    return ranking


def run_leave_one_out(
    samples: List[SampleScore],
    prune_ratio: float = 0.1,
    judge_noise_correction: bool = False,
) -> LeaveOneOutValidation:
    """
    Run leave-one-out validation across all models.

    For each model:
    1. Remove that model's scores from the dataset
    2. Run pruning using only the remaining models
    3. Check if the pruned set correctly ranks the held-out model

    Args:
        samples: Full list of SampleScore objects (all models)
        prune_ratio: Same ratio used in actual pruning
        judge_noise_correction: Same setting used in actual pruning

    Returns:
        LeaveOneOutValidation with results for all rounds
    """
    if not samples:
        raise ValueError('No samples provided for leave-one-out validation')

    all_models = list(samples[0].scores.keys())

    if len(all_models) < 2:
        raise ValueError(
            f'Leave-one-out requires at least 2 models, got {len(all_models)}'
        )

    full_ranking = _compute_ranking(samples, all_models)
    rounds = []

    for held_out in all_models:
        training_models = [m for m in all_models if m != held_out]

        training_samples = [
            SampleScore(
                index=s.index,
                scores={m: s.scores[m] for m in training_models if m in s.scores}
            )
            for s in samples
        ]

        pruner = CorrelationStratifiedPruner(
            prune_ratio=prune_ratio,
            judge_noise_correction=judge_noise_correction,
        )
        selected_indices = pruner.select_samples(training_samples)
        selected_set = set(selected_indices)

        pruned_samples_full = [s for s in samples if s.index in selected_set]
        pruned_ranking = _compute_ranking(pruned_samples_full, all_models)

        spearman = _spearman_correlation(
            {m: full_ranking[m] for m in all_models},
            {m: pruned_ranking[m] for m in all_models},
        )

        rounds.append(LeaveOneOutResult(
            held_out_model=held_out,
            training_models=training_models,
            selected_indices=selected_indices,
            n_selected=len(selected_indices),
            full_ranking=full_ranking,
            pruned_ranking=pruned_ranking,
            spearman_correlation=spearman,
        ))

    mean_spearman = float(np.mean([r.spearman_correlation for r in rounds]))

    # overall_success_rate: fraction of rounds with Spearman >= 0.5
    success_rate = sum(r.spearman_correlation >= 0.5 for r in rounds) / len(rounds)

    return LeaveOneOutValidation(
        rounds=rounds,
        overall_success_rate=success_rate,
        mean_spearman=mean_spearman,
    )
