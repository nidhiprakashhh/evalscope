"""
Universal mixin that adds correlation-stratified sample filtering to any
evalscope benchmark adapter.

Usage:
    class MyPrunedAdapter(UniversalPrunedAdapterMixin, MyBaseAdapter):
        def _compute_pruned_indices(self) -> Optional[set]:
            # benchmark-specific pruning logic
            ...

The mixin owns the full sample_filter / record_to_sample / get_pruning_stats
lifecycle. Subclasses only need to implement _compute_pruned_indices().

For adapters where sample identity cannot be read from the record (e.g. MMMU,
which lacks a stable per-row index), override _get_pruner_index(record) to
return a deterministic index value instead.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from evalscope.api.dataset import Sample

# 4 levels up from evalscope_ext/pruning/ reaches the project root,
# where Evals/ lives alongside task1/ and task2/.
_DEFAULT_EVALS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'Evals')
)


class UniversalPrunedAdapterMixin:
    """
    Universal mixin that adds correlation-stratified sample filtering to any
    evalscope benchmark adapter. Works across LCB, AA-LCR, MMMU, and any
    future benchmark registered via BenchmarkMeta — no per-benchmark
    adapter boilerplate needed.

    Subclasses must implement _compute_pruned_indices(). Everything else is
    handled here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pruned_indices: Optional[set] = None
        self._pruning_stats: Optional[Dict] = None
        self._pruning_attempted: bool = False

    def _get_evals_dir(self) -> str:
        """Resolve path to the Evals/ directory."""
        evals_dir = (
            self.extra_params.get('evals_dir')
            or os.environ.get('EVALS_DIR')
            or _DEFAULT_EVALS_DIR
        )
        return os.path.abspath(evals_dir)

    def _compute_pruned_indices(self) -> Optional[set]:
        """
        Return the set of sample indices to keep, or None to use the full
        benchmark (no pruning). Called once on the first sample_filter call.

        Subclasses must implement this.
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement _compute_pruned_indices()'
        )

    def _get_dataset_param(self, key: str, default=None):
        """
        Resolve a parameter with priority:
        1. task_config.dataset_args top-level (spec format:
           --dataset-args '{"pruning_strategy": "...", "prune_ratio": 0.1}')
        2. self.extra_params (nested format:
           --dataset-args '{"benchmark_name": {"extra_params": {...}}}')
        3. default

        The two formats are mutually exclusive in practice — the nested format
        stores keys under the benchmark name, not at top level — so checking
        top-level first works correctly for both.
        """
        if self._task_config is not None:
            top_level_val = self._task_config.dataset_args.get(key)
            if top_level_val is not None:
                return top_level_val
        val = self.extra_params.get(key)
        if val is not None:
            return val
        return default

    def _get_pruner_index(self, record: Dict[str, Any]) -> Any:
        """
        Extract the pruner index from a record. Default: use the record's
        'index' or 'id' field.

        Override in subclasses where records lack a stable index field
        (e.g. MMMU — override to use a sample counter instead).
        """
        return record.get('index') or record.get('id')

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert record to Sample, injecting _pruner_index into metadata."""
        sample = super().record_to_sample(record)
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata['_pruner_index'] = self._get_pruner_index(record)
        return sample

    def sample_filter(self, sample: Sample) -> bool:
        """
        Filter samples to only include pruned indices.
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
        """Return pruning statistics populated by _compute_pruned_indices."""
        return self._pruning_stats
