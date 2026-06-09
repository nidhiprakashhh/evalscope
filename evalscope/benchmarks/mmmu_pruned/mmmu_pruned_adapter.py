"""
MMMU pruned dataset adapter.

Registers as 'mmmu_pruned' in evalscope's benchmark registry.
Uses difficulty stratification and visual complexity scoring
since only one model is available in the reference data
(cross-model variance is not computable).

Part A: Prunes 660 reference samples for eval pipeline.
Part B: encoder probe lives in evalscope_ext/tools/mmmu_probe.py

Usage:
    evalscope eval --model <model> \\
        --datasets mmmu_pruned \\
        --dataset-args '{"prune_ratio": 0.2}' \\
        --output ./results_pruned/

    # Encoder probe mode (Part B):
    evalscope eval --model <model> \\
        --datasets mmmu_pruned \\
        --dataset-args '{"prune_ratio": 0.2, "encoder_probe_mode": true}' \\
        --output ./results_pruned/
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter, SUBSET_LIST, OPEN_PROMPT
from evalscope.utils.logger import get_logger

from evalscope_ext.pruning.universal_pruned_adapter import UniversalPrunedAdapterMixin
from evalscope_ext.pruning.mmmu_pruner import (
    MmmuPruner,
    load_mmmu_samples,
)

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_pruned',
        pretty_name='MMMU (Pruned)',
        dataset_id='AI-ModelScope/MMMU',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        subset_list=SUBSET_LIST,
        description=(
            'MMMU with difficulty-stratified pruning and visual complexity '
            'scoring. Selects encoder-stressing samples across all subjects.'
        ),
        metric_list=['acc'],
        eval_split='validation',
        prompt_template=OPEN_PROMPT,
        extra_params={
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to keep (default 0.2).',
                'value': 0.2,
            },
            'evals_dir': {
                'type': 'str | null',
                'description': 'Path to Evals/ directory.',
                'value': None,
            },
            'encoder_probe_mode': {
                'type': 'bool',
                'description': (
                    'If True, select encoder probe set targeting '
                    'image encoder weaknesses specifically (Part B).'
                ),
                'value': False,
            },
        },
    )
)
class MMMUPrunedAdapter(UniversalPrunedAdapterMixin, MMMUAdapter):
    """
    Pruned variant of MMMUAdapter.

    Inherits sample_filter / get_pruning_stats from PrunedAdapterMixin.
    Overrides record_to_sample to use a counter-based index (MMMU records
    lack a stable per-row index field), and _get_pruner_index accordingly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_counter: int = 0

    def _get_pruner_index(self, record: Dict[str, Any]) -> int:
        """MMMU records have no stable index — use a sequential counter instead."""
        idx = self._sample_counter
        self._sample_counter += 1
        if idx <= 3 or idx % 30 == 0:
            logger.debug(f'MMMU adapter sample counter: {idx}')
        return idx

    def _compute_pruned_indices(self) -> Optional[set]:
        """
        Load MMMU per-sample scores from Evals/MMMU/ and select the most
        encoder-stressful samples using difficulty stratification and
        visual complexity scoring.
        """
        prune_ratio = float(self._get_dataset_param('prune_ratio', 0.2))
        encoder_probe_mode = bool(
            self._get_dataset_param('encoder_probe_mode', False)
        )

        evals_dir = self._get_evals_dir()
        reviews_dir = os.path.join(
            evals_dir, 'MMMU', 'reviews', 'glm-4.5v-fp8'
        )
        predictions_dir = os.path.join(
            evals_dir, 'MMMU', 'predictions', 'glm-4.5v-fp8'
        )

        if not os.path.exists(reviews_dir):
            logger.warning(
                f'MMMU reviews directory not found at {reviews_dir}. '
                f'Falling back to full benchmark. '
                f'Set EVALS_DIR environment variable to enable pruning.'
            )
            return None

        try:
            samples = load_mmmu_samples(
                predictions_dir=predictions_dir,
                reviews_dir=reviews_dir,
            )

            if not samples:
                logger.warning('No MMMU samples loaded. Using full benchmark.')
                return None

            pruner = MmmuPruner(prune_ratio=prune_ratio)

            selected_indices = pruner.select_samples(samples)
            self._pruning_stats = pruner.get_pruning_stats(
                samples, selected_indices
            )

            mode_label = 'encoder probe' if encoder_probe_mode else 'standard'
            logger.info(
                f'MMMU pruning ({mode_label}): selected '
                f'{len(selected_indices)} / {len(samples)} samples '
                f'(prune_ratio={prune_ratio})'
            )

            return set(selected_indices)

        except Exception as e:
            logger.error(f'MMMU pruning failed: {e}. Using full benchmark.')
            return None
