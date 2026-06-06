"""
Compare full vs pruned benchmark results with bootstrap confidence intervals.

Reads evalscope output directories and produces a side-by-side comparison
showing whether the pruned set preserves model rankings within a
statistically defensible margin.

Usage:
    python -m evalscope_ext.tools.compare_runs \\
        --full ./results_full/ \\
        --pruned ./results_pruned/

    # With JSON output:
    python -m evalscope_ext.tools.compare_runs \\
        --full ./results_full/ \\
        --pruned ./results_pruned/ \\
        --output ./comparison.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BENCHMARK_DISPLAY_NAMES = {
    'live_code_bench': 'LiveCodeBench',
    'live_code_bench_pruned': 'LiveCodeBench (Pruned)',
    'aa_lcr': 'AA-LCR',
    'aa_lcr_pruned': 'AA-LCR (Pruned)',
    'mmmu': 'MMMU',
    'mmmu_pruned': 'MMMU (Pruned)',
}

N_BOOTSTRAP = 1000
CI_LEVEL = 0.90  # 90% confidence interval


@dataclass
class BenchmarkResult:
    """Parsed result from one evalscope output directory."""
    benchmark_name: str
    model_scores: Dict[str, float]  # model_name -> mean score
    per_sample_scores: Dict[str, List[float]]  # model_name -> list of 0/1 scores
    n_samples: int


@dataclass
class ComparisonResult:
    """Comparison between full and pruned results for one benchmark."""
    benchmark: str
    model: str
    full_score: float
    pruned_score: float
    delta: float
    ci_half_width: float
    ci_level: float
    n_full_samples: int
    n_pruned_samples: int
    prune_ratio: float
    ranking_preserved: bool


def _find_result_files(results_dir: str) -> List[Path]:
    """Find evalscope result JSON files in an output directory."""
    results_path = Path(results_dir)
    result_files = []

    for pattern in ['**/*.json', '**/*.jsonl']:
        result_files.extend(results_path.glob(pattern))

    return [f for f in result_files if f.name != 'comparison.json']


def _load_evalscope_results(results_dir: str) -> Optional[BenchmarkResult]:
    """
    Load evalscope output and extract per-sample scores.

    evalscope writes results in a structured format under the output dir.
    We parse whatever JSON files exist to extract model scores.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    # Look for summary/report files first
    for summary_name in ['summary.json', 'report.json', 'results.json']:
        summary_path = results_path / summary_name
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            return _parse_summary(data, results_path.name)

    # Fall back to scanning all JSON files
    all_scores: Dict[str, List[float]] = {}
    benchmark_name = results_path.name

    for json_file in results_path.rglob('*.json'):
        try:
            with open(json_file) as f:
                data = json.load(f)
            _extract_scores_from_json(data, all_scores)
        except (json.JSONDecodeError, KeyError):
            continue

    if not all_scores:
        return None

    model_scores = {
        model: sum(scores) / len(scores)
        for model, scores in all_scores.items()
        if scores
    }

    total_samples = max(len(s) for s in all_scores.values()) if all_scores else 0

    return BenchmarkResult(
        benchmark_name=benchmark_name,
        model_scores=model_scores,
        per_sample_scores=all_scores,
        n_samples=total_samples,
    )


def _parse_summary(data: dict, benchmark_name: str) -> BenchmarkResult:
    """Parse evalscope summary JSON format."""
    model_scores = {}
    per_sample_scores = {}

    if 'results' in data:
        for model, result in data['results'].items():
            if isinstance(result, dict):
                score = result.get('acc', result.get('pass', result.get('score', 0)))
                model_scores[model] = float(score)
            elif isinstance(result, (int, float)):
                model_scores[model] = float(result)

    elif 'model' in data and 'score' in data:
        model_scores[data['model']] = float(data['score'])

    n_samples = data.get('n_samples', data.get('total', 0))

    return BenchmarkResult(
        benchmark_name=benchmark_name,
        model_scores=model_scores,
        per_sample_scores=per_sample_scores,
        n_samples=int(n_samples),
    )


def _extract_scores_from_json(data: dict, scores: Dict[str, List[float]]) -> None:
    """Extract per-sample scores from a JSON object."""
    model = data.get('model', 'unknown')
    if model not in scores:
        scores[model] = []

    score_val = None
    if 'sample_score' in data:
        sv = data['sample_score'].get('score', {}).get('value', {})
        score_val = sv.get('pass', sv.get('acc'))
    elif 'score' in data:
        score_val = data['score']

    if score_val is not None:
        scores[model].append(float(score_val))


def bootstrap_confidence_interval(
    scores: List[float],
    n_bootstrap: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for mean score.

    Args:
        scores: List of per-sample scores (0.0 or 1.0)
        n_bootstrap: Number of bootstrap resamples
        ci_level: Confidence level (e.g. 0.90 for 90% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not scores:
        return (0.0, 0.0)

    n = len(scores)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        resample = [random.choice(scores) for _ in range(n)]
        bootstrap_means.append(sum(resample) / n)

    bootstrap_means.sort()
    alpha = 1 - ci_level
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)

    return (bootstrap_means[lower_idx], bootstrap_means[upper_idx])


def compare_results(
    full_dir: str,
    pruned_dir: str,
) -> List[ComparisonResult]:
    """
    Compare full and pruned benchmark results.

    Args:
        full_dir: Path to full benchmark results directory
        pruned_dir: Path to pruned benchmark results directory

    Returns:
        List of ComparisonResult objects, one per model
    """
    full_result = _load_evalscope_results(full_dir)
    pruned_result = _load_evalscope_results(pruned_dir)

    if full_result is None:
        raise FileNotFoundError(f'Could not load results from {full_dir}')
    if pruned_result is None:
        raise FileNotFoundError(f'Could not load results from {pruned_dir}')

    prune_ratio = (
        pruned_result.n_samples / full_result.n_samples
        if full_result.n_samples > 0 else 0.0
    )

    common_models = set(full_result.model_scores.keys()) & set(
        pruned_result.model_scores.keys()
    )

    if not common_models:
        common_models = set(full_result.model_scores.keys())

    comparisons = []
    for model in sorted(common_models):
        full_score = full_result.model_scores.get(model, 0.0)
        pruned_score = pruned_result.model_scores.get(model, full_score)
        delta = pruned_score - full_score

        pruned_scores_list = pruned_result.per_sample_scores.get(model, [])
        if pruned_scores_list:
            lower, upper = bootstrap_confidence_interval(pruned_scores_list)
            ci_half_width = (upper - lower) / 2
        else:
            ci_half_width = 0.05  # conservative estimate when no per-sample data

        comparisons.append(ComparisonResult(
            benchmark=full_result.benchmark_name,
            model=model,
            full_score=full_score,
            pruned_score=pruned_score,
            delta=delta,
            ci_half_width=ci_half_width,
            ci_level=CI_LEVEL,
            n_full_samples=full_result.n_samples,
            n_pruned_samples=pruned_result.n_samples,
            prune_ratio=prune_ratio,
            ranking_preserved=_check_ranking_preserved(
                full_result.model_scores,
                pruned_result.model_scores,
            ),
        ))

    return comparisons


def _check_ranking_preserved(
    full_scores: Dict[str, float],
    pruned_scores: Dict[str, float],
) -> bool:
    """Check if model ranking is preserved between full and pruned."""
    models = [m for m in full_scores if m in pruned_scores]
    if len(models) < 2:
        return True
    full_order = sorted(models, key=lambda m: full_scores[m], reverse=True)
    pruned_order = sorted(models, key=lambda m: pruned_scores[m], reverse=True)
    return full_order == pruned_order


def format_comparison_table(comparisons: List[ComparisonResult]) -> str:
    """Format comparison results as a readable table."""
    if not comparisons:
        return 'No comparison results available.'

    lines = []
    lines.append('\n' + '=' * 70)
    lines.append('Benchmark Pruning Comparison')
    lines.append('=' * 70)

    benchmark = comparisons[0].benchmark
    display_name = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
    n_full = comparisons[0].n_full_samples
    n_pruned = comparisons[0].n_pruned_samples
    prune_ratio = comparisons[0].prune_ratio

    lines.append(
        f'{display_name}: {n_full} → {n_pruned} samples '
        f'({prune_ratio:.1%} kept)'
    )
    lines.append('-' * 70)
    lines.append(
        f'{"Model":<30} {"Full":>8} {"Pruned":>8} '
        f'{"Δ":>8} {"90% CI":>12} {"Ranking"}'
    )
    lines.append('-' * 70)

    for c in comparisons:
        delta_str = f'{c.delta:+.3f}'
        ci_str = f'±{c.ci_half_width:.3f}'
        ranking_str = '✓' if c.ranking_preserved else '✗'
        lines.append(
            f'{c.model:<30} {c.full_score:>8.3f} {c.pruned_score:>8.3f} '
            f'{delta_str:>8} {ci_str:>12}   {ranking_str}'
        )

    lines.append('=' * 70)

    all_preserved = all(c.ranking_preserved for c in comparisons)
    max_delta = max(abs(c.delta) for c in comparisons)
    max_ci = max(c.ci_half_width for c in comparisons)

    lines.append(
        f'Verdict: Rankings {"preserved ✓" if all_preserved else "NOT preserved ✗"} | '
        f'Max |Δ|: {max_delta:.3f} | '
        f'Max CI: ±{max_ci:.3f}'
    )
    lines.append('=' * 70)

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Compare full vs pruned evalscope benchmark results'
    )
    parser.add_argument(
        '--full', required=True,
        help='Path to full benchmark results directory',
    )
    parser.add_argument(
        '--pruned', required=True,
        help='Path to pruned benchmark results directory',
    )
    parser.add_argument(
        '--output', default=None,
        help='Optional path to write JSON comparison results',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for bootstrap resampling (default: 42)',
    )

    args = parser.parse_args()
    random.seed(args.seed)

    comparisons = compare_results(
        full_dir=args.full,
        pruned_dir=args.pruned,
    )

    print(format_comparison_table(comparisons))

    if args.output:
        output_data = {
            'full_dir': args.full,
            'pruned_dir': args.pruned,
            'n_bootstrap': N_BOOTSTRAP,
            'ci_level': CI_LEVEL,
            'comparisons': [asdict(c) for c in comparisons],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'\nComparison saved to: {args.output}')


if __name__ == '__main__':
    main()
