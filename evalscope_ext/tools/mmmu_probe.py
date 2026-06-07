"""
MMMU Encoder Stress Coverage Probe — Part B implementation.

Selects a ~150-sample probe set from the full ~12K HuggingFace MMMU dataset
to surface image encoder degradation across 5 stress categories.

Approach
--------
1. Load all 30 MMMU subjects from HuggingFace
2. Assign each sample a stress category via img_type metadata (authoritative)
3. Compute pixel-level stress score for within-category ranking (numpy/scipy):
   - Edge density (Sobel filter): structural line complexity
   - Grayscale entropy: information density
   - Layout complexity (regional std): spatial non-uniformity
   - Text region likelihood (thresholding): text vs. visual content
4. Require >=2 signals above 0.5 threshold to classify as high-stress
5. Allocate target budget across 5 clusters; redistribute unused budget
   proportionally to clusters with more candidates
6. Evaluate using MMMU's real questions and ground-truth answers

Stress categories
-----------------
  tables        Structured grids requiring precise cell reading
  dense_text    Screenshots and text-heavy images
  charts        Plots, bar/line charts, trees, maps
  diagrams      Blueprints, chemical structures, geometric shapes
  fine_grained  Microscopic/medical images, detail-rich photographs

Selection uses zero model outputs -- generalizes to unseen models.

Usage
-----
    # Select probe set
    python -m evalscope_ext.tools.mmmu_probe \\
        --mode select \\
        --target-size 150 \\
        --output ./mmmu_probe_set.json

    # Report per-category accuracy vs reference model
    python -m evalscope_ext.tools.mmmu_probe \\
        --mode report \\
        --probe-file ./mmmu_probe_set.json \\
        --results-dir ./results_pruned/ \\
        --reference-dir ./Evals/MMMU/reviews/glm-4.5v-fp8/ \\
        --output ./mmmu_probe_report.json
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from PIL import Image, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from scipy.ndimage import sobel as scipy_sobel
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# Default Evals directory — same resolution pattern as mmmu_pruned_adapter.py,
# adjusted for this file's location (4 levels up to ai-model-quality-challenge/).
DEFAULT_EVALS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..', 'Evals',
))


# ── Category mapping ───────────────────────────────────────────────────────────

# img_type metadata is the authoritative signal for category assignment.
# Pixel features are used only for within-category stress ranking.
# Unmapped img_type values return None and are skipped entirely.
IMG_TYPE_TO_CATEGORY: Dict[str, str] = {
    # tables
    'Tables':                                'tables',
    # dense_text
    'Screenshots':                           'dense_text',
    'Comics and Cartoons':                   'dense_text',
    'Mathematical Notations':                'dense_text',
    'DNA Sequences':                         'dense_text',
    'Poster':                                'dense_text',
    'Advertisements':                        'dense_text',
    # charts
    'Plots and Charts':                      'charts',
    'Trees and Graphs':                      'charts',
    'Maps':                                  'charts',
    # diagrams
    'Diagrams':                              'diagrams',
    'Technical Blueprints':                  'diagrams',
    'Chemical Structures':                   'diagrams',
    'Geometric Shapes':                      'diagrams',
    'Scientific Figures':                    'diagrams',
    'Sketches and Drafts':                   'diagrams',
    'Icons and Symbols':                     'diagrams',
    # fine_grained
    'Microscopic Images':                    'fine_grained',
    'Pathological Images':                   'fine_grained',
    'Body Scans: MRI, CT scans, and X-rays': 'fine_grained',
    'Medical Images':                        'fine_grained',
    'Portraits':                             'fine_grained',
    'Photographs':                           'fine_grained',
    'Paintings':                             'fine_grained',
    'Logos and Branding':                    'fine_grained',
    # unmapped: Landscapes, Sculpture, Other -> None (skipped)
}

ALL_CATEGORIES = ('tables', 'dense_text', 'charts', 'diagrams', 'fine_grained')

MMMU_SUBJECTS = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art',
    'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry',
    'Clinical_Medicine', 'Computer_Science', 'Design',
    'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
    'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature',
    'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering',
    'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology',
]


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class ProbeRecord:
    """A single sample selected for the encoder probe set."""
    index: int                  # sequential counter across all HF subjects
    hf_id: str                  # HF record id (e.g. 'validation_Accounting_1')
    subject: str
    subfield: str
    topic_difficulty: str
    category: str               # one of ALL_CATEGORIES
    img_types: List[str]        # raw MMMU img_type values
    stress_score: float
    features: Dict[str, float]  # raw signal values
    question: str
    answer: str
    choices: Dict[str, str]


@dataclass
class ProbeSet:
    """Selected encoder probe set with selection metadata."""
    records: List[ProbeRecord]
    total_hf_samples: int
    target_size: int
    category_allocation: Dict[str, int]  # budget per category
    category_selected: Dict[str, int]    # actual count selected per category
    mean_stress_score: float


# ── Image helpers ──────────────────────────────────────────────────────────────

def _load_hf_image(raw: Any) -> Optional['Image.Image']:
    """Load PIL Image from an HF record image field (PIL Image, dict, or bytes)."""
    if not PIL_AVAILABLE:
        return None
    try:
        if isinstance(raw, Image.Image):
            return raw.copy()
        if isinstance(raw, dict) and 'bytes' in raw:
            return Image.open(io.BytesIO(raw['bytes']))
        return Image.open(io.BytesIO(raw))
    except Exception:
        return None


def _parse_img_type(raw: Any) -> List[str]:
    """Parse img_type field which may be a string repr of a list."""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            result = ast.literal_eval(raw)
            return [str(x) for x in result] if isinstance(result, list) else [raw]
        except (ValueError, SyntaxError):
            return [raw] if raw else []
    return []


def _resolve_category(img_types: List[str]) -> Optional[str]:
    """
    Map img_type list to a stress category.
    Returns None if no img_type maps to a known category — caller skips sample.
    """
    for t in img_types:
        cat = IMG_TYPE_TO_CATEGORY.get(t)
        if cat is not None:
            return cat
    return None


# ── Pixel feature extraction ───────────────────────────────────────────────────

def _compute_stress_features(img: 'Image.Image') -> Dict[str, float]:
    """
    Compute 4 pixel-level stress signals from a PIL image.
    All signals normalized to [0, 1].
    Resizes to max 512px before processing for memory efficiency.
    """
    resample = getattr(getattr(Image, 'Resampling', None), 'LANCZOS', None)
    if resample is None:
        resample = getattr(Image, 'LANCZOS', getattr(Image, 'ANTIALIAS', 1))
    img.thumbnail((512, 512), resample)

    gray = np.array(img.convert('L'), dtype=np.float32)
    h, w = gray.shape

    # 1. Edge density via Sobel.
    # Normalized by 50 (not 255) so typical edge-dense images (mean Sobel ~12-15)
    # produce scores in the 0.25-0.30 range rather than near-zero.
    if SCIPY_AVAILABLE:
        sx = scipy_sobel(gray, axis=0)
        sy = scipy_sobel(gray, axis=1)
        edge_mag = np.hypot(sx, sy)
    else:
        edge_arr = np.array(
            img.convert('L').filter(ImageFilter.FIND_EDGES),
            dtype=np.float32,
        )
        edge_mag = edge_arr
    edge_density = min(float(np.mean(edge_mag)) / 50.0, 1.0)

    # 2. Grayscale entropy (information density); normalized by log2(256) = 8
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0.0, 256.0))
    hist_p = hist / (hist.sum() + 1e-10)
    nz = hist_p[hist_p > 0]
    entropy = min(float(-np.sum(nz * np.log2(nz))) / 8.0, 1.0)

    # 3. Layout complexity: std of 3x3 regional mean intensities
    region_means = [
        float(np.mean(gray[i * h // 3:(i + 1) * h // 3,
                           j * w // 3:(j + 1) * w // 3]))
        for i in range(3)
        for j in range(3)
        if gray[i * h // 3:(i + 1) * h // 3,
                j * w // 3:(j + 1) * w // 3].size > 0
    ]
    layout_complexity = (
        min(float(np.std(region_means)) / 128.0, 1.0) if region_means else 0.0
    )

    # 4. Text region likelihood: fraction of near-black or near-white pixels
    text_likelihood = min(float(np.mean((gray < 50) | (gray > 200))), 1.0)

    return {
        'edge_density':      edge_density,
        'entropy':           entropy,
        'layout_complexity': layout_complexity,
        'text_likelihood':   text_likelihood,
    }


def _stress_score(features: Dict[str, float]) -> float:
    """
    Combine signals into a single stress score.
    Requires >=2 signals above 0.5 for full weight; otherwise halved
    (single-signal agreement may reflect noise rather than genuine stress).
    """
    vals = list(features.values())
    base = float(np.mean(vals))
    n_high = sum(v > 0.5 for v in vals)
    return base if n_high >= 2 else base * 0.5


# ── Budget allocation ──────────────────────────────────────────────────────────

def _allocate_budget(
    cluster_sizes: Dict[str, int],
    total_budget: int,
) -> Dict[str, int]:
    """
    Distribute total_budget across clusters.

    Base = total // n_clusters, capped at cluster size.
    Unused budget redistributed proportionally to clusters with more candidates.
    Never leaves budget unused if any cluster can absorb it.
    """
    n = len(cluster_sizes)
    if n == 0:
        return {}

    base = total_budget // n
    allocation = {cat: min(base, sz) for cat, sz in cluster_sizes.items()}
    remaining = total_budget - sum(allocation.values())

    if remaining > 0:
        can_absorb = {
            cat: cluster_sizes[cat] - allocation[cat]
            for cat in cluster_sizes
            if cluster_sizes[cat] > allocation[cat]
        }
        total_absorb = sum(can_absorb.values())
        if total_absorb > 0:
            for cat, absorb in can_absorb.items():
                allocation[cat] += int(remaining * absorb / total_absorb)
            leftover = total_budget - sum(allocation.values())
            if leftover > 0:
                biggest = max(can_absorb, key=lambda c: can_absorb[c])
                allocation[biggest] += leftover

    return allocation


# ── Selection ──────────────────────────────────────────────────────────────────

def select_probe_set(
    target_size: int = 150,
    hf_split: str = 'validation',
    subjects: Optional[List[str]] = None,
) -> ProbeSet:
    """
    Load HuggingFace MMMU, score every sample, select top-stress representatives.

    Samples with img_type values not in IMG_TYPE_TO_CATEGORY are skipped.

    Args:
        target_size: Total probe set size (default 150)
        hf_split: HF dataset split to load (default 'validation')
        subjects: Override subject list (default: all 30 MMMU subjects)

    Returns:
        ProbeSet with selected records and selection statistics
    """
    if not HF_AVAILABLE:
        raise ImportError('datasets required: pip install datasets')
    if not PIL_AVAILABLE:
        raise ImportError('Pillow required: pip install Pillow')

    load_subjects = subjects or MMMU_SUBJECTS

    # category -> list of (stress_score, ProbeRecord)
    candidates: Dict[str, list] = defaultdict(list)
    global_idx = 0
    total_loaded = 0
    total_skipped = 0

    print(f'Loading {len(load_subjects)} MMMU subjects ({hf_split} split)...')

    for subject in load_subjects:
        try:
            ds = load_dataset('MMMU/MMMU', subject, split=hf_split)
        except Exception as e:
            print(f'  Warning: could not load {subject}: {e}')
            continue

        subject_count = 0
        subject_skipped = 0
        for record in ds:
            img_types = _parse_img_type(record.get('img_type', []))
            category = _resolve_category(img_types)

            if category is None:
                # img_type not in any known stress category — skip
                total_skipped += 1
                subject_skipped += 1
                global_idx += 1
                continue

            # Compute stress score from first available image
            score = 0.5
            features: Dict[str, float] = {}
            for img_key in ('image_1', 'image_2', 'image_3'):
                raw = record.get(img_key)
                if raw is None:
                    continue
                img = _load_hf_image(raw)
                if img is not None:
                    try:
                        features = _compute_stress_features(img)
                        score = _stress_score(features)
                    except Exception:
                        pass
                    break

            choices = {
                k: str(record[k])
                for k in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
                if record.get(k)
            }

            rec = ProbeRecord(
                index=global_idx,
                hf_id=str(record.get('id', f'{subject}_{global_idx}')),
                subject=subject,
                subfield=str(record.get('subfield', '')),
                topic_difficulty=str(record.get('topic_difficulty', 'Medium')),
                category=category,
                img_types=img_types,
                stress_score=round(score, 4),
                features={k: round(v, 4) for k, v in features.items()},
                question=str(record.get('question', '')),
                answer=str(record.get('answer', '')),
                choices=choices,
            )
            candidates[category].append((score, rec))
            global_idx += 1
            subject_count += 1

        total_loaded += subject_count
        print(f'  {subject}: {subject_count} kept, {subject_skipped} skipped')

    print(f'\nTotal: {total_loaded} kept, {total_skipped} skipped '
          f'(unmapped img_type)')

    for cat in candidates:
        candidates[cat].sort(key=lambda x: x[0], reverse=True)

    cluster_sizes = {cat: len(candidates.get(cat, [])) for cat in ALL_CATEGORIES}
    allocation = _allocate_budget(cluster_sizes, target_size)

    print(f'\nBudget allocation (target={target_size}):')
    for cat in ALL_CATEGORIES:
        print(f'  {cat:<15}: {allocation.get(cat, 0):3d} selected '
              f'from {cluster_sizes.get(cat, 0)} candidates')

    selected: List[ProbeRecord] = []
    category_selected: Dict[str, int] = {}
    for cat in ALL_CATEGORIES:
        n = allocation.get(cat, 0)
        chosen = [rec for _, rec in candidates.get(cat, [])[:n]]
        selected.extend(chosen)
        category_selected[cat] = len(chosen)

    mean_stress = float(np.mean([r.stress_score for r in selected])) if selected else 0.0

    return ProbeSet(
        records=selected,
        total_hf_samples=total_loaded,
        target_size=target_size,
        category_allocation=dict(allocation),
        category_selected=category_selected,
        mean_stress_score=round(mean_stress, 4),
    )


# ── Reference scores ───────────────────────────────────────────────────────────

def load_reference_scores(
    reference_dir: str,
    probe_records: List[ProbeRecord],
) -> Dict[str, float]:
    """
    Compute per-category accuracy for a reference model on the same probe samples.

    Reads per-subject review JSONL files (mmmu_<Subject>.jsonl) from a reference
    model's review directory (e.g. Evals/MMMU/reviews/glm-4.5v-fp8/).

    Join key: row['sample_score']['sample_metadata']['id'] == record.hf_id

    Returns:
        {category: accuracy} for categories with at least one matched sample.
    """
    ref_path = Path(reference_dir)
    review_files = sorted(ref_path.glob('mmmu_*.jsonl'))
    if not review_files:
        raise FileNotFoundError(
            f'No reference review files found in {reference_dir}. '
            f'Expected mmmu_<Subject>.jsonl files.'
        )

    id_to_score: Dict[str, float] = {}
    for review_file in review_files:
        with open(review_file) as f:
            for line in f:
                row = json.loads(line.strip())
                hf_id = (
                    row.get('sample_score', {})
                       .get('sample_metadata', {})
                       .get('id')
                )
                if not hf_id:
                    continue
                score_val = (
                    row.get('sample_score', {})
                       .get('score', {})
                       .get('value', {})
                       .get('acc')
                )
                if score_val is not None:
                    id_to_score[hf_id] = float(score_val)

    cat_correct: Dict[str, float] = defaultdict(float)
    cat_total: Dict[str, int] = defaultdict(int)
    for record in probe_records:
        score = id_to_score.get(record.hf_id)
        if score is None:
            continue
        cat_correct[record.category] += score
        cat_total[record.category] += 1

    return {
        cat: round(cat_correct[cat] / cat_total[cat], 4)
        for cat in cat_total
        if cat_total[cat] > 0
    }


# ── Accuracy report ────────────────────────────────────────────────────────────

def compute_category_accuracy(
    probe_set: ProbeSet,
    results_dir: str,
    reference_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute per-category accuracy from evalscope results.

    Join key: row['sample_score']['sample_metadata']['id'] == record.hf_id

    If reference_scores provided, adds delta and go/no-go verdict per category.
    If not, reports raw scores with a note recommending reference comparison.

    Args:
        probe_set: Selected probe set
        results_dir: evalscope results directory (contains mmmu_pruned__*.jsonl)
        reference_scores: Optional {category: accuracy} from load_reference_scores()

    Returns:
        Report dict with per-category accuracy and optional verdict.
    """
    results_path = Path(results_dir)
    review_files = list(results_path.rglob('mmmu_pruned__*.jsonl'))
    if not review_files:
        review_files = list(results_path.rglob('mmmu*.jsonl'))
    if not review_files:
        raise FileNotFoundError(
            f'No MMMU results found in {results_dir}. '
            f'Expected: mmmu_pruned__<model>.jsonl'
        )

    id_to_score: Dict[str, float] = {}
    for review_file in review_files:
        with open(review_file) as f:
            for line in f:
                row = json.loads(line.strip())
                hf_id = (
                    row.get('sample_score', {})
                       .get('sample_metadata', {})
                       .get('id')
                )
                if not hf_id:
                    continue
                score_val = (
                    row.get('sample_score', {})
                       .get('score', {})
                       .get('value', {})
                       .get('acc')
                )
                if score_val is not None:
                    id_to_score[hf_id] = float(score_val)

    cat_correct: Dict[str, float] = defaultdict(float)
    cat_total: Dict[str, int] = defaultdict(int)
    matched = 0
    for record in probe_set.records:
        score = id_to_score.get(record.hf_id)
        if score is None:
            continue
        cat_correct[record.category] += score
        cat_total[record.category] += 1
        matched += 1

    category_accuracy: Dict[str, Dict] = {}
    for cat in ALL_CATEGORIES:
        total = cat_total[cat]
        correct = cat_correct[cat]
        acc = round(correct / total, 4) if total > 0 else None

        entry: Dict[str, Any] = {
            'accuracy': acc,
            'correct':  int(correct),
            'total':    total,
        }

        if reference_scores is not None:
            ref = reference_scores.get(cat)
            if acc is not None and ref is not None:
                delta = round(acc - ref, 4)

                # Thresholds are relative to a reference model (glm-4.5v-fp8).
                # No absolute encoder accuracy thresholds exist in published
                # literature — MMMU overall scores (56-87% across model tiers)
                # mix reasoning and visual extraction. Relative comparison
                # against a known reference is more meaningful for encoder
                # quality screening. A delta > 15% below reference on any
                # category warrants investigation before full MMMU evaluation.
                if delta >= -0.05:
                    verdict = 'PASS'
                elif delta >= -0.15:
                    verdict = 'REVIEW'
                else:
                    verdict = 'FAIL'

                entry['reference'] = ref
                entry['delta']     = delta
                entry['verdict']   = verdict
            else:
                entry['reference'] = ref
                entry['delta']     = None
                entry['verdict']   = 'insufficient_data'
        else:
            entry['verdict'] = 'no_reference'

        category_accuracy[cat] = entry

    overall_total = sum(cat_total.values())
    return {
        'total_probe_records': len(probe_set.records),
        'matched_results':     matched,
        'reference_provided':  reference_scores is not None,
        'overall_accuracy':    (
            round(sum(cat_correct.values()) / overall_total, 4)
            if overall_total > 0 else None
        ),
        'category_accuracy':   category_accuracy,
    }


# ── Output formatting ──────────────────────────────────────────────────────────

def _print_probe_summary(probe_set: ProbeSet) -> None:
    print('\n' + '=' * 55)
    print('MMMU Encoder Stress Probe — Selection Summary')
    print('=' * 55)
    print(f'HF samples kept:     {probe_set.total_hf_samples}')
    print(f'Probe set size:      {len(probe_set.records)} '
          f'(target: {probe_set.target_size})')
    print(f'Mean stress score:   {probe_set.mean_stress_score:.3f}')
    print('\nPer-category:')
    for cat in ALL_CATEGORIES:
        alloc  = probe_set.category_allocation.get(cat, 0)
        actual = probe_set.category_selected.get(cat, 0)
        print(f'  {cat:<15}: {actual:3d} selected  (budget: {alloc})')
    print('=' * 55)


def _print_report_summary(report: Dict[str, Any]) -> None:
    print('\n' + '=' * 65)
    print('MMMU Encoder Stress Probe — Accuracy Report')
    print('=' * 65)
    print(f'Probe records:    {report["total_probe_records"]}')
    print(f'Matched results:  {report["matched_results"]} / {report["total_probe_records"]}')
    print(f'Overall accuracy: {report.get("overall_accuracy", "N/A")}')

    has_ref = report.get('reference_provided', False)
    if has_ref:
        print(f'\n  {"Category":<13}  {"Score":>6}  {"Ref":>6}  {"Delta":>7}  Verdict')
        print('  ' + '-' * 51)
        for cat, stats in report['category_accuracy'].items():
            acc     = stats['accuracy']
            ref     = stats.get('reference')
            delta   = stats.get('delta')
            verdict = stats.get('verdict', '')
            if acc is not None and ref is not None and delta is not None:
                print(f'  {cat:<13}  {acc:>6.3f}  {ref:>6.3f}  {delta:>+7.3f}  {verdict}')
            else:
                print(f'  {cat:<13}  {"N/A":>6}  {"N/A":>6}  {"N/A":>7}  insufficient_data')
    else:
        print('\n  Note: no reference model provided.')
        print('  Pass --reference-dir for delta/verdict comparison (recommended).')
        print(f'\n  {"Category":<13}  {"Score":>6}')
        print('  ' + '-' * 22)
        for cat, stats in report['category_accuracy'].items():
            acc = stats['accuracy']
            print(f'  {cat:<13}  {acc:>6.3f}' if acc is not None else
                  f'  {cat:<13}  {"N/A":>6}')
    print('=' * 65)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _default_reference_dir() -> Optional[str]:
    """
    Resolve the default reference directory using the same EVALS_DIR pattern
    as mmmu_pruned_adapter.py: env var > DEFAULT_EVALS_DIR.
    """
    evals_dir = os.environ.get('EVALS_DIR') or DEFAULT_EVALS_DIR
    candidate = os.path.join(
        os.path.abspath(evals_dir), 'MMMU', 'reviews', 'glm-4.5v-fp8'
    )
    return candidate if os.path.isdir(candidate) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description='MMMU Encoder Stress Coverage Probe'
    )
    parser.add_argument(
        '--mode', choices=['select', 'report'], required=True,
        help=(
            'select: score and select probe set from HuggingFace MMMU. '
            'report: compute per-category accuracy from evalscope results.'
        ),
    )
    parser.add_argument(
        '--target-size', type=int, default=150,
        help='Total probe set size (default: 150)',
    )
    parser.add_argument(
        '--hf-split', type=str, default='validation',
        help='HuggingFace dataset split (default: validation)',
    )
    parser.add_argument(
        '--probe-file', type=str, default=None,
        help='Probe set JSON file (required for report mode)',
    )
    parser.add_argument(
        '--results-dir', type=str, default=None,
        help='evalscope results directory (required for report mode)',
    )
    parser.add_argument(
        '--reference-dir', type=str, default=None,
        help=(
            'Reference model review directory for delta/verdict comparison. '
            'Default: auto-detect via EVALS_DIR env var or DEFAULT_EVALS_DIR.'
        ),
    )
    parser.add_argument(
        '--output', type=str, default='./mmmu_probe_set.json',
        help='Output JSON file path',
    )

    args = parser.parse_args()

    if args.mode == 'select':
        probe_set = select_probe_set(
            target_size=args.target_size,
            hf_split=args.hf_split,
        )
        _print_probe_summary(probe_set)

        output_data = {
            'mode':                'select',
            'total_hf_samples':    probe_set.total_hf_samples,
            'target_size':         probe_set.target_size,
            'mean_stress_score':   probe_set.mean_stress_score,
            'category_allocation': probe_set.category_allocation,
            'category_selected':   probe_set.category_selected,
            'records':             [asdict(r) for r in probe_set.records],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'\nProbe set saved to: {args.output}')

    elif args.mode == 'report':
        if not args.probe_file:
            parser.error('--probe-file required for report mode')
        if not args.results_dir:
            parser.error('--results-dir required for report mode')

        with open(args.probe_file) as f:
            data = json.load(f)

        probe_records = [
            ProbeRecord(
                index=r['index'],
                hf_id=r['hf_id'],
                subject=r['subject'],
                subfield=r['subfield'],
                topic_difficulty=r['topic_difficulty'],
                category=r['category'],
                img_types=r['img_types'],
                stress_score=r['stress_score'],
                features=r['features'],
                question=r['question'],
                answer=r['answer'],
                choices=r['choices'],
            )
            for r in data['records']
        ]
        probe_set = ProbeSet(
            records=probe_records,
            total_hf_samples=data['total_hf_samples'],
            target_size=data['target_size'],
            category_allocation=data['category_allocation'],
            category_selected=data['category_selected'],
            mean_stress_score=data['mean_stress_score'],
        )

        # Resolve reference directory: CLI arg > EVALS_DIR env > DEFAULT_EVALS_DIR
        ref_dir = args.reference_dir or _default_reference_dir()
        reference_scores: Optional[Dict[str, float]] = None
        if ref_dir:
            try:
                reference_scores = load_reference_scores(ref_dir, probe_records)
                print(f'Reference scores loaded from: {ref_dir}')
            except FileNotFoundError as e:
                print(f'Warning: {e}')
                print('Proceeding without reference comparison.')
        else:
            print('No reference directory found. '
                  'Pass --reference-dir or set EVALS_DIR for delta/verdict output.')

        report = compute_category_accuracy(
            probe_set, args.results_dir, reference_scores
        )
        _print_report_summary(report)

        out = (
            args.output
            if args.output != './mmmu_probe_set.json'
            else str(Path(args.probe_file).parent / 'mmmu_probe_report.json')
        )
        with open(out, 'w') as f:
            json.dump({**report, 'probe_file': args.probe_file}, f, indent=2)
        print(f'\nReport saved to: {out}')


if __name__ == '__main__':
    main()
