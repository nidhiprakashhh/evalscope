"""
AA-LCR pruned dataset adapter.

Registers as 'aa_lcr_pruned' in evalscope's benchmark registry.
Applies correlation-stratified pruning with judge noise correction
to select the most discriminating subset of AA-LCR samples.

AA-LCR uses an LLM judge (non-deterministic). We apply cross-model
consistency weighting to down-weight samples where score patterns
may reflect judge noise rather than genuine capability differences.

Usage:
    evalscope eval --model <model> \\
        --datasets aa_lcr_pruned \\
        --dataset-args '{"pruning_strategy": "correlation_stratified",
                         "prune_ratio": 0.2}' \\
        --output ./results_pruned/
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
from evalscope.utils.logger import get_logger

from evalscope_ext.pruning.correlation_stratified import (
    CorrelationStratifiedPruner,
    load_scores_from_jsonl,
)
from evalscope_ext.pruning.leave_one_out import run_leave_one_out

logger = get_logger()

DEFAULT_EVALS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..', '..', 'Evals'
))


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        dataset_id='evalscope/AA-LCR',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description=(
            'AA-LCR with correlation-stratified pruning and judge noise '
            'correction. Selects the most reliable discriminating samples.'
        ),
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning algorithm to use.',
                'value': 'correlation_stratified',
            },
            'prune_ratio': {
                'type': 'float',
                'description': (
                    'Fraction of samples to keep. Default 0.2 (higher than LCB) '
                    'to account for LLM judge noise in AA-LCR scoring.'
                ),
                'value': 0.2,
            },
            'evals_dir': {
                'type': 'str | null',
                'description': 'Path to Evals/ directory.',
                'value': None,
            },
            'run_validation': {
                'type': 'bool',
                'description': 'Run leave-one-out validation after pruning.',
                'value': False,
            },
            'text_dir': {
                'type': 'str | null',
                'description': 'Local directory containing extracted AA-LCR text files.',
                'value': None,
            },
        },
    )
)
class AALCRPrunedAdapter(AALCRAdapter):
    """
    Pruned variant of AALCRAdapter.

    Inherits all AA-LCR evaluation logic. Uses record_to_sample injection
    to pass sample indices to sample_filter for pruning.

    Key difference from LCB: judge_noise_correction=True to handle
    non-deterministic LLM judge scoring in AA-LCR.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pruned_indices: Optional[set] = None
        self._pruning_stats: Optional[Dict] = None
        self._pruning_attempted: bool = False

    def _get_evals_dir(self) -> str:
        evals_dir = (
            self.extra_params.get('evals_dir')
            or os.environ.get('EVALS_DIR')
            or DEFAULT_EVALS_DIR
        )
        return os.path.abspath(evals_dir)

    def _compute_pruned_indices(self) -> Optional[set]:
        """
        Compute which sample indices to keep using correlation-stratified
        pruning with judge noise correction.
        """
        strategy = self.extra_params.get('pruning_strategy', 'correlation_stratified')
        prune_ratio = float(self.extra_params.get('prune_ratio', 0.2))
        run_validation = bool(self.extra_params.get('run_validation', False))
        logger.info(f'AA-LCR pruning strategy: {strategy}')

        evals_dir = self._get_evals_dir()
        predictions_dir = os.path.join(evals_dir, 'Part 1', 'predictions')
        reviews_dir = os.path.join(evals_dir, 'Part 1', 'reviews')

        if not os.path.exists(reviews_dir):
            logger.warning(
                f'Evals reviews directory not found at {reviews_dir}. '
                f'Falling back to full benchmark. '
                f'Set EVALS_DIR environment variable to enable pruning.'
            )
            return None

        try:
            samples = load_scores_from_jsonl(
                predictions_dir=predictions_dir,
                reviews_dir=reviews_dir,
                benchmark_prefix='aa_lcr',
                score_key='acc',
            )

            if not samples:
                logger.warning('No AA-LCR samples loaded. Using full benchmark.')
                return None

            pruner = CorrelationStratifiedPruner(
                prune_ratio=prune_ratio,
                judge_noise_correction=True,  # AA-LCR uses LLM judge
            )

            selected_indices = pruner.select_samples(samples)
            self._pruning_stats = pruner.get_pruning_stats(
                samples, selected_indices
            )

            logger.info(
                f'AA-LCR pruning: selected {len(selected_indices)} / '
                f'{len(samples)} samples (prune_ratio={prune_ratio})'
            )
            logger.info(
                f'Ranking preserved: '
                f'{self._pruning_stats["ranking_preserved"]}'
            )

            if run_validation:
                logger.info('Running leave-one-out validation...')
                validation = run_leave_one_out(
                    samples=samples,
                    prune_ratio=prune_ratio,
                    judge_noise_correction=True,
                )
                logger.info('\n' + validation.summary())

            return set(selected_indices)

        except Exception as e:
            logger.error(
                f'AA-LCR pruning failed: {e}. Using full benchmark.'
            )
            return None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Inject record index into metadata for sample_filter."""
        sample = super().record_to_sample(record)
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata['_pruner_index'] = record.get('index') or record.get('id')
        return sample

    def sample_filter(self, sample: Sample) -> bool:
        """Filter to pruned indices only."""
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
