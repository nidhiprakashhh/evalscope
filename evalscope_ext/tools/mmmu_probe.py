"""
MMMU Encoder Probe Tool — Part B implementation.

Selects a probe set from the full 12K HuggingFace MMMU dataset
specifically designed to surface image encoder degradation.

Unlike generic benchmark sampling, this probe targets samples where:
1. The question REQUIRES visual parsing to answer correctly
2. The image type specifically stresses encoder capabilities
3. Subjects are diverse to prevent domain bias

Also implements perturbation testing methodology:
- Divide image into 3x3 grid
- Blur each patch independently
- Measure score degradation per patch via OpenAI API
- Encoder failure = score drops on blur, not on question rephrasing

Usage:
    # Select probe set from full 12K HF dataset
    python -m evalscope_ext.tools.mmmu_probe \\
        --mode select \\
        --prune-ratio 0.05 \\
        --output ./mmmu_probe_results.json

    # Run perturbation test on selected probe set
    python -m evalscope_ext.tools.mmmu_probe \\
        --mode perturb \\
        --probe-file ./mmmu_probe_results.json \\
        --model gpt-4o \\
        --api-key <your-key> \\
        --output ./mmmu_perturbation_results.json
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# PIL for image perturbation
try:
    from PIL import Image, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# HuggingFace datasets for full 12K MMMU
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# OpenAI client for perturbation testing
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from evalscope_ext.pruning.mmmu_pruner import (
    ENCODER_STRESSING_TYPES,
    VISUALLY_DEPENDENT_SUBJECTS,
    VISUAL_COMPLEXITY,
)


# ── Probe set selection ────────────────────────────────────────────────────────

@dataclass
class ProbeRecord:
    """A single record selected for the encoder probe set."""
    index: int
    subject: str
    subfield: str
    topic_difficulty: str
    img_types: List[str]
    visual_complexity: float
    question: str
    answer: str
    choices: Optional[Dict[str, str]] = None


@dataclass
class ProbeSet:
    """Selected encoder probe set with metadata."""
    records: List[ProbeRecord]
    total_hf_samples: int
    selection_ratio: float
    subject_distribution: Dict[str, int]
    difficulty_distribution: Dict[str, int]
    img_type_distribution: Dict[str, int]


def _parse_img_type(raw: Any) -> List[str]:
    """Parse img_type field which may be a string repr of a list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [raw] if raw else []
    return []


def _visual_complexity_score(img_types: List[str]) -> float:
    """Compute max visual complexity across img_types."""
    if not img_types:
        return 0.3
    return max(VISUAL_COMPLEXITY.get(t, 0.3) for t in img_types)


def select_probe_set(
    prune_ratio: float = 0.05,
    hf_split: str = 'validation',
    min_per_subject: int = 3,
) -> ProbeSet:
    """
    Select encoder probe set from full 12K HuggingFace MMMU dataset.

    Strategy:
    1. Load all subjects from HF
    2. Filter to visually-dependent subjects
    3. Score each sample by visual complexity
    4. Guarantee min_per_subject coverage
    5. Fill remaining budget with highest-complexity samples

    Args:
        prune_ratio: Fraction of total samples to select
        hf_split: HuggingFace dataset split to use
        min_per_subject: Minimum samples per subject in probe set

    Returns:
        ProbeSet with selected records and statistics
    """
    if not HF_AVAILABLE:
        raise ImportError(
            'datasets library required. Install with: pip install datasets'
        )

    print(f'Loading MMMU {hf_split} split from HuggingFace...')
    all_records = []

    for subject in sorted(VISUALLY_DEPENDENT_SUBJECTS):
        try:
            ds = load_dataset('MMMU/MMMU', subject, split=hf_split)
            for i, record in enumerate(ds):
                img_types = _parse_img_type(record.get('img_type', []))
                complexity = _visual_complexity_score(img_types)
                all_records.append({
                    'index': len(all_records),
                    'subject': subject,
                    'subfield': record.get('subfield', ''),
                    'topic_difficulty': record.get('topic_difficulty', 'Medium'),
                    'img_types': img_types,
                    'visual_complexity': complexity,
                    'question': record.get('question', ''),
                    'answer': record.get('answer', ''),
                    'choices': {
                        k: record[k]
                        for k in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                        if k in record and record[k]
                    },
                    '_raw': record,
                })
            print(f'  Loaded {len(ds)} samples from {subject}')
        except Exception as e:
            print(f'  Warning: could not load {subject}: {e}')

    if not all_records:
        raise RuntimeError('No MMMU records loaded from HuggingFace')

    print(f'\nTotal records loaded: {len(all_records)}')

    total_budget = max(
        len(VISUALLY_DEPENDENT_SUBJECTS) * min_per_subject,
        int(len(all_records) * prune_ratio),
    )
    print(f'Target probe set size: {total_budget} '
          f'({prune_ratio:.1%} of {len(all_records)})')

    selected = []
    covered_subjects = defaultdict(int)

    # Phase 1: subject coverage
    by_subject = defaultdict(list)
    for r in all_records:
        by_subject[r['subject']].append(r)

    for subject, subject_records in by_subject.items():
        subject_records.sort(key=lambda r: r['visual_complexity'], reverse=True)
        for r in subject_records[:min_per_subject]:
            selected.append(r)
            covered_subjects[r['subject']] += 1

    # Phase 2: fill remaining by visual complexity
    selected_indices = {r['index'] for r in selected}
    remaining = total_budget - len(selected)

    if remaining > 0:
        candidates = [
            r for r in all_records
            if r['index'] not in selected_indices
        ]
        candidates.sort(key=lambda r: r['visual_complexity'], reverse=True)
        selected.extend(candidates[:remaining])

    # Build ProbeSet
    probe_records = [
        ProbeRecord(
            index=r['index'],
            subject=r['subject'],
            subfield=r['subfield'],
            topic_difficulty=r['topic_difficulty'],
            img_types=r['img_types'],
            visual_complexity=r['visual_complexity'],
            question=r['question'],
            answer=r['answer'],
            choices=r.get('choices'),
        )
        for r in selected
    ]

    return ProbeSet(
        records=probe_records,
        total_hf_samples=len(all_records),
        selection_ratio=len(probe_records) / len(all_records),
        subject_distribution=dict(Counter(r.subject for r in probe_records)),
        difficulty_distribution=dict(Counter(r.topic_difficulty for r in probe_records)),
        img_type_distribution=dict(
            Counter(t for r in probe_records for t in r.img_types).most_common(10)
        ),
    )


# ── Perturbation testing ───────────────────────────────────────────────────────

@dataclass
class PatchResult:
    """Score degradation for a single image patch."""
    patch_row: int      # 0-2 (top to bottom)
    patch_col: int      # 0-2 (left to right)
    original_score: float
    blurred_score: float
    degradation: float  # original - blurred (positive = encoder was using this patch)


@dataclass
class PerturbationResult:
    """Full perturbation result for one probe sample."""
    index: int
    subject: str
    question: str
    original_score: float
    patch_results: List[PatchResult]
    max_degradation: float
    most_important_patch: Tuple[int, int]  # (row, col)
    encoder_dependent: bool  # True if any patch causes >0.3 degradation


def _blur_patch(
    img: 'Image.Image',
    row: int,
    col: int,
    blur_radius: int = 15,
) -> 'Image.Image':
    """
    Return a copy of img with one 3x3 grid patch blurred.

    Args:
        img: Original PIL image
        row: Grid row (0=top, 1=middle, 2=bottom)
        col: Grid column (0=left, 1=center, 2=right)
        blur_radius: GaussianBlur radius

    Returns:
        New PIL image with the specified patch blurred
    """
    w, h = img.size
    patch_w, patch_h = w // 3, h // 3

    x1 = col * patch_w
    y1 = row * patch_h
    x2 = x1 + patch_w if col < 2 else w
    y2 = y1 + patch_h if row < 2 else h

    blurred = img.copy()
    patch = blurred.crop((x1, y1, x2, y2))
    patch = patch.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred.paste(patch, (x1, y1))
    return blurred


def _image_to_base64(img: 'Image.Image') -> str:
    """Convert PIL image to base64 string for OpenAI API."""
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _query_model(
    client: 'OpenAI',
    model: str,
    question: str,
    choices: Optional[Dict[str, str]],
    image_b64: str,
    correct_answer: str,
) -> float:
    """
    Query model with image and question via OpenAI API.
    Returns 1.0 if correct, 0.0 if wrong, -1.0 if API call failed.
    """
    if choices:
        choices_text = '\n'.join(f'{k}) {v}' for k, v in choices.items())
        prompt = (
            f'{question}\n\n{choices_text}\n\n'
            f'Answer with the letter only (e.g. A, B, C...).'
        )
    else:
        prompt = question

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/jpeg;base64,{image_b64}'
                            },
                        },
                        {'type': 'text', 'text': prompt},
                    ],
                }
            ],
            max_tokens=10,
            temperature=0,
        )
        prediction = response.choices[0].message.content.strip().upper()
        return 1.0 if prediction.startswith(correct_answer.upper()) else 0.0
    except Exception as e:
        print(f'    API error: {e}')
        return -1.0  # sentinel for failed call


def run_perturbation_test(
    probe_records: List[ProbeRecord],
    hf_split: str,
    model: str,
    api_key: str,
    max_samples: int = 20,
    blur_radius: int = 15,
    rate_limit_delay: float = 1.0,
) -> List[PerturbationResult]:
    """
    Run 3x3 grid perturbation test on probe samples.

    For each sample:
    1. Get original score (no blur)
    2. For each of 9 patches: blur patch, get score, measure degradation
    3. Identify which patches the encoder relies on

    Args:
        probe_records: Selected probe samples
        hf_split: HF split to load images from
        model: OpenAI model name (e.g. 'gpt-4o')
        api_key: OpenAI API key
        max_samples: Maximum samples to test (API cost control)
        blur_radius: GaussianBlur radius for patch blurring
        rate_limit_delay: Seconds to wait between API calls

    Returns:
        List of PerturbationResult objects
    """
    if not PIL_AVAILABLE:
        raise ImportError('Pillow required. Install with: pip install Pillow')
    if not OPENAI_AVAILABLE:
        raise ImportError('openai required. Install with: pip install openai')
    if not HF_AVAILABLE:
        raise ImportError('datasets required. Install with: pip install datasets')

    client = OpenAI(api_key=api_key)
    results = []

    # Sample subset for cost control
    test_records = probe_records[:max_samples]
    subjects_needed = set(r.subject for r in test_records)

    # Load HF images for needed subjects
    print(f'Loading images for {len(subjects_needed)} subjects...')
    subject_datasets = {}
    for subject in subjects_needed:
        try:
            ds = load_dataset(
                'MMMU/MMMU', subject, split=hf_split
            )
            subject_datasets[subject] = {row['id']: row for row in ds}
        except Exception as e:
            print(f'  Warning: could not load {subject}: {e}')

    print(f'\nRunning perturbation test on {len(test_records)} samples...')

    for i, record in enumerate(test_records):
        print(f'  Sample {i+1}/{len(test_records)}: {record.subject} '
              f'(idx={record.index})')

        # Get raw HF record for image
        subject_data = subject_datasets.get(record.subject, {})
        if not subject_data:
            print(f'    Skipping — subject data not loaded')
            continue

        # Find matching record by question text
        hf_record = None
        for hf_row in subject_data.values():
            if hf_row.get('question', '').strip() == record.question.strip():
                hf_record = hf_row
                break

        if hf_record is None:
            print(f'    Skipping — could not match record in HF dataset')
            continue

        # Get first image
        raw_image = None
        for img_key in ['image_1', 'image_2', 'image_3']:
            if hf_record.get(img_key) is not None:
                raw_image = hf_record[img_key]
                break

        if raw_image is None:
            print(f'    Skipping — no image found')
            continue

        if isinstance(raw_image, Image.Image):
            img = raw_image
        elif isinstance(raw_image, dict) and 'bytes' in raw_image:
            img = Image.open(io.BytesIO(raw_image['bytes']))
        else:
            img = Image.open(io.BytesIO(raw_image))

        # Step 1: original score
        original_b64 = _image_to_base64(img)
        original_score = _query_model(
            client, model, record.question, record.choices,
            original_b64, record.answer
        )
        time.sleep(rate_limit_delay)

        if original_score < 0:
            print(f'    Skipping — original query failed')
            continue

        print(f'    Original score: {original_score}')

        # Step 2: perturb each patch
        patch_results = []
        for row in range(3):
            for col in range(3):
                blurred_img = _blur_patch(img, row, col, blur_radius)
                blurred_b64 = _image_to_base64(blurred_img)
                blurred_score = _query_model(
                    client, model, record.question, record.choices,
                    blurred_b64, record.answer
                )
                time.sleep(rate_limit_delay)

                degradation = (
                    original_score - blurred_score
                    if blurred_score >= 0 else 0.0
                )
                patch_results.append(PatchResult(
                    patch_row=row,
                    patch_col=col,
                    original_score=original_score,
                    blurred_score=blurred_score if blurred_score >= 0 else original_score,
                    degradation=degradation,
                ))
                print(f'    Patch ({row},{col}): degradation={degradation:.2f}')

        if not patch_results:
            continue

        best_patch = max(patch_results, key=lambda p: p.degradation)
        max_deg = best_patch.degradation

        results.append(PerturbationResult(
            index=record.index,
            subject=record.subject,
            question=record.question[:100],
            original_score=original_score,
            patch_results=patch_results,
            max_degradation=max_deg,
            most_important_patch=(best_patch.patch_row, best_patch.patch_col),
            encoder_dependent=max_deg > 0.3,
        ))

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def _print_probe_summary(probe_set: ProbeSet) -> None:
    print('\n' + '=' * 50)
    print('MMMU Encoder Probe Set — Summary')
    print('=' * 50)
    print(f'Total HF samples scanned:  {probe_set.total_hf_samples}')
    print(f'Probe set size:            {len(probe_set.records)}')
    print(f'Selection ratio:           {probe_set.selection_ratio:.2%}')
    print(f'\nDifficulty distribution:')
    for diff, count in sorted(probe_set.difficulty_distribution.items()):
        print(f'  {diff}: {count}')
    print(f'\nTop image types:')
    for img_type, count in list(probe_set.img_type_distribution.items())[:5]:
        print(f'  {img_type}: {count}')
    print(f'\nSubjects covered: {len(probe_set.subject_distribution)}')
    print('=' * 50)


def _print_perturbation_summary(results: List[PerturbationResult]) -> None:
    print('\n' + '=' * 50)
    print('Perturbation Test Results')
    print('=' * 50)
    encoder_dependent = [r for r in results if r.encoder_dependent]
    print(f'Samples tested:            {len(results)}')
    print(f'Encoder-dependent:         {len(encoder_dependent)} '
          f'({len(encoder_dependent)/len(results):.1%})')
    print(f'\nMost impactful patches:')
    patch_counts = Counter(r.most_important_patch for r in results)
    for patch, count in patch_counts.most_common(3):
        print(f'  Patch {patch}: {count} samples rely on this region')
    print('=' * 50)


def main():
    parser = argparse.ArgumentParser(
        description='MMMU Encoder Probe Tool — Part B implementation'
    )
    parser.add_argument(
        '--mode',
        choices=['select', 'perturb'],
        required=True,
        help='select: choose probe set from HF. perturb: run perturbation test.',
    )
    parser.add_argument(
        '--prune-ratio', type=float, default=0.05,
        help='Fraction of HF samples to select for probe set (default: 0.05)',
    )
    parser.add_argument(
        '--probe-file', type=str, default=None,
        help='Path to probe set JSON (required for perturb mode)',
    )
    parser.add_argument(
        '--model', type=str, default='gpt-4o',
        help='OpenAI model for perturbation testing',
    )
    parser.add_argument(
        '--api-key', type=str, default=None,
        help='OpenAI API key (or set OPENAI_API_KEY env var)',
    )
    parser.add_argument(
        '--max-samples', type=int, default=20,
        help='Max samples to perturb (API cost control, default: 20)',
    )
    parser.add_argument(
        '--output', type=str, default='./mmmu_probe_results.json',
        help='Output JSON file path',
    )
    parser.add_argument(
        '--hf-split', type=str, default='validation',
        help='HuggingFace dataset split to use',
    )

    args = parser.parse_args()

    if args.mode == 'select':
        probe_set = select_probe_set(
            prune_ratio=args.prune_ratio,
            hf_split=args.hf_split,
        )
        _print_probe_summary(probe_set)

        output_data = {
            'mode': 'select',
            'total_hf_samples': probe_set.total_hf_samples,
            'selection_ratio': probe_set.selection_ratio,
            'subject_distribution': probe_set.subject_distribution,
            'difficulty_distribution': probe_set.difficulty_distribution,
            'img_type_distribution': probe_set.img_type_distribution,
            'records': [asdict(r) for r in probe_set.records],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'\nProbe set saved to: {args.output}')

    elif args.mode == 'perturb':
        if not args.probe_file:
            parser.error('--probe-file required for perturb mode')

        api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            parser.error('--api-key or OPENAI_API_KEY env var required')

        with open(args.probe_file) as f:
            probe_data = json.load(f)

        probe_records = [ProbeRecord(**r) for r in probe_data['records']]

        results = run_perturbation_test(
            probe_records=probe_records,
            hf_split=args.hf_split,
            model=args.model,
            api_key=api_key,
            max_samples=args.max_samples,
        )

        _print_perturbation_summary(results)

        output_data = {
            'mode': 'perturb',
            'model': args.model,
            'samples_tested': len(results),
            'results': [asdict(r) for r in results],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'\nPerturbation results saved to: {args.output}')


if __name__ == '__main__':
    main()
