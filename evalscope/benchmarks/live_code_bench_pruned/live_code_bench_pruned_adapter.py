"""
LiveCodeBench pruned dataset adapter.

Registers as 'live_code_bench_pruned' in evalscope's benchmark registry.
Applies correlation-stratified pruning to select the most discriminating
subset of LCB samples.

Usage:
    evalscope eval --model <model> \\
        --datasets live_code_bench_pruned \\
        --dataset-args '{"pruning_strategy": "correlation_stratified",
                         "prune_ratio": 0.1}' \\
        --output ./results_pruned/
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import (
    LiveCodeBenchAdapter,
)
from evalscope.utils.logger import get_logger

from evalscope_ext.pruning.universal_pruned_adapter import UniversalPrunedAdapterMixin
from evalscope_ext.pruning.correlation_stratified import (
    CorrelationStratifiedPruner,
    load_scores_from_jsonl,
)
from evalscope_ext.pruning.leave_one_out import run_leave_one_out

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (Pruned)',
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        tags=[Tags.CODING],
        subset_list=[
            'release_latest',
            'release_v1',
            'release_v2',
            'release_v3',
            'release_v4',
            'release_v5',
            'release_v6',
            'v1',
            'v1_v2',
            'v1_v3',
            'v1_v4',
            'v1_v5',
            'v1_v6',
            'v2',
            'v2_v3',
            'v2_v4',
            'v2_v5',
            'v2_v6',
            'v3',
            'v3_v4',
            'v3_v5',
            'v3_v6',
            'v4',
            'v4_v5',
            'v4_v6',
            'v5',
            'v5_v6',
            'v6',
        ],
        description=(
            'LiveCodeBench v5 with correlation-stratified pruning. '
            'Selects the smallest sample set that preserves model rankings.'
        ),
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template=(
            '### Question:\n{question_content}\n\n'
            '{format_prompt} ### Answer: (use the provided format with backticks)\n\n'
        ),
        review_timeout=6,
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning algorithm to use.',
                'value': 'correlation_stratified',
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to keep (e.g. 0.1 = 10%).',
                'value': 0.1,
            },
            'evals_dir': {
                'type': 'str | null',
                'description': 'Path to Evals/ directory. Falls back to EVALS_DIR env var.',
                'value': None,
            },
            'run_validation': {
                'type': 'bool',
                'description': 'Run leave-one-out validation after pruning.',
                'value': False,
            },
        },
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {}
            }
        },
    )
)
class LiveCodeBenchPrunedAdapter(UniversalPrunedAdapterMixin, LiveCodeBenchAdapter):
    """
    Pruned variant of LiveCodeBenchAdapter.

    Inherits sample_filter / record_to_sample / get_pruning_stats from
    PrunedAdapterMixin. Only implements _compute_pruned_indices() with
    LCB-specific data loading and pruner configuration.
    """

    def _compute_pruned_indices(self) -> Optional[set]:
        """
        Load LCB per-sample scores from Evals/Part 1/ and select the
        most discriminating subset using correlation-stratified pruning.
        """
        strategy = self._get_dataset_param('pruning_strategy', 'correlation_stratified')
        prune_ratio = float(self._get_dataset_param('prune_ratio', 0.1))
        run_validation = bool(self._get_dataset_param('run_validation', False))
        logger.info(f'LCB pruning strategy: {strategy}')

        evals_dir = self._get_evals_dir()
        predictions_dir = os.path.join(evals_dir, 'Part 1', 'predictions')
        reviews_dir = os.path.join(evals_dir, 'Part 1', 'reviews')

        if not os.path.exists(reviews_dir):
            logger.warning(
                f'Evals reviews directory not found at {reviews_dir}. '
                f'Falling back to full benchmark (no pruning). '
                f'Set EVALS_DIR environment variable to enable pruning.'
            )
            return None

        try:
            samples = load_scores_from_jsonl(
                predictions_dir=predictions_dir,
                reviews_dir=reviews_dir,
                benchmark_prefix='live_code_bench_v5',
                score_key='pass',
            )

            if not samples:
                logger.warning('No LCB samples loaded. Using full benchmark.')
                return None

            pruner = CorrelationStratifiedPruner(
                prune_ratio=prune_ratio,
                judge_noise_correction=False,  # LCB uses deterministic grader
            )

            selected_indices = pruner.select_samples(samples)
            self._pruning_stats = pruner.get_pruning_stats(samples, selected_indices)

            logger.info(
                f'LCB pruning: selected {len(selected_indices)} / '
                f'{len(samples)} samples (prune_ratio={prune_ratio})'
            )
            logger.info(
                f'Ranking preserved: {self._pruning_stats["ranking_preserved"]}'
            )

            if run_validation:
                logger.info('Running leave-one-out validation...')
                validation = run_leave_one_out(
                    samples=samples,
                    prune_ratio=prune_ratio,
                    judge_noise_correction=False,
                )
                logger.info('\n' + validation.summary())

            return set(selected_indices)

        except Exception as e:
            logger.error(f'LCB pruning failed: {e}. Using full benchmark.')
            return None
