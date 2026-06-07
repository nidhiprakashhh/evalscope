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
from typing import Any, Dict, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import (
    LiveCodeBenchAdapter,
)
from evalscope.utils.logger import get_logger

from evalscope_ext.pruning.correlation_stratified import (
    CorrelationStratifiedPruner,
    load_scores_from_jsonl,
)
from evalscope_ext.pruning.leave_one_out import run_leave_one_out

logger = get_logger()

# Default path to Evals data — override via EVALS_DIR env var
DEFAULT_EVALS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..', '..', 'Evals'
))


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (Pruned)',
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        tags=[Tags.CODING],
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
    )
)
class LiveCodeBenchPrunedAdapter(LiveCodeBenchAdapter):
    """
    Pruned variant of LiveCodeBenchAdapter.

    Inherits all LCB evaluation logic and overrides record_to_sample()
    to inject the raw record index into sample metadata, then overrides
    sample_filter() to apply correlation-stratified pruning.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pruned_indices: Optional[set] = None
        self._pruning_stats: Optional[Dict] = None
        self._pruning_attempted: bool = False

    def _get_evals_dir(self) -> str:
        """Resolve path to Evals directory."""
        evals_dir = (
            self.extra_params.get('evals_dir')
            or os.environ.get('EVALS_DIR')
            or DEFAULT_EVALS_DIR
        )
        return os.path.abspath(evals_dir)

    def _compute_pruned_indices(self) -> Optional[set]:
        """
        Compute which sample indices to keep.
        Called once on first sample_filter invocation.
        """
        strategy = self.extra_params.get('pruning_strategy', 'correlation_stratified')
        prune_ratio = float(self.extra_params.get('prune_ratio', 0.1))
        run_validation = bool(self.extra_params.get('run_validation', False))
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

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert record to Sample, injecting index into metadata for filtering."""
        sample = super().record_to_sample(record)
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata['_pruner_index'] = record.get('index') or record.get('id')
        return sample

    def sample_filter(self, sample: Sample) -> bool:
        """
        Filter samples to only include pruned indices.

        Called by evalscope for each sample during dataset loading.
        Returns True to keep, False to skip.
        """
        if not self._pruning_attempted:
            self._pruning_attempted = True
            self._pruned_indices = self._compute_pruned_indices()

        if self._pruned_indices is None:
            return True

        idx = sample.metadata.get('_pruner_index') if sample.metadata else None
        if idx is None:
            return True

        return int(idx) in self._pruned_indices

    def get_pruning_stats(self) -> Optional[Dict]:
        """Return pruning statistics for reporting."""
        return self._pruning_stats
