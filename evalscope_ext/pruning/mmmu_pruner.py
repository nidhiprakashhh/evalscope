"""
MMMU-specific pruner for single-model benchmark compression.

Cannot use cross-model discrimination (only 1 model in shipped data).
Instead stratifies by topic_difficulty and visual complexity (img_type).

Selection strategy:
1. Reserve budget for subject diversity (min 1 per subject)
2. Allocate remaining budget by difficulty (Easy 20%, Medium 50%, Hard 30%)
3. Within each difficulty bin, rank by visual complexity score

NOTE: This is Part A of the MMMU implementation. Part B (full 12K encoder
probe) is in evalscope_ext/tools/mmmu_probe.py.
"""

from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# Visual complexity scores by img_type.
# Higher = more encoder-dependent, more valuable for stress-testing an image encoder.
# Chemical Structures require precise bond geometry; Photographs do not.
VISUAL_COMPLEXITY: Dict[str, float] = {
    'Chemical Structures': 1.00,
    'Technical Blueprints': 0.95,
    'Microscopic Images': 0.90,
    'Pathological Images': 0.88,
    'Geometric Shapes': 0.85,
    'Body Scans: MRI, CT scans, and X-rays': 0.82,
    'Medical Images': 0.80,
    'Diagrams': 0.75,
    'Scientific Figures': 0.60,
    'Plots and Charts': 0.58,
    'Trees and Graphs': 0.55,
    'Maps': 0.50,
    'Tables': 0.40,
    'Photographs': 0.35,
    'Paintings': 0.30,
    'Portraits': 0.28,
    'Sketches and Drafts': 0.25,
    'Comics and Cartoons': 0.20,
    'Screenshots': 0.15,
    'Other': 0.30,
}

ENCODER_STRESSING_TYPES = {
    'Diagrams',
    'Chemical Structures',
    'Technical Blueprints',
    'Geometric Shapes',
    'Medical Images',
    'Microscopic Images',
    'Pathological Images',
    'Body Scans: MRI, CT scans, and X-rays',
    'Plots and Charts',
}

VISUALLY_DEPENDENT_SUBJECTS = {
    'Clinical_Medicine',
    'Electronics',
    'Architecture_and_Engineering',
    'Chemistry',
    'Basic_Medical_Science',
    'Diagnostics_and_Laboratory_Medicine',
    'Materials',
    'Energy_and_Power',
    'Biology',
    'Art',
}

# Budget allocation per difficulty bin (must sum to 1.0)
DEFAULT_DIFFICULTY_WEIGHTS: Dict[str, float] = {
    'Easy': 0.20,
    'Medium': 0.50,
    'Hard': 0.30,
}


@dataclass
class MmmuSample:
    """A single MMMU sample with metadata and score."""
    index: int
    subject: str           # MMMU subject (e.g. 'Accounting')
    subfield: str          # finer grain (e.g. 'Investment')
    topic_difficulty: str  # 'Easy', 'Medium', or 'Hard'
    img_type: List[str]    # e.g. ['Diagrams', 'Tables']
    score: float           # 0.0 or 1.0 from single model

    @property
    def visual_complexity(self) -> float:
        """Max visual complexity score across all img_types for this sample."""
        if not self.img_type:
            return 0.30
        return max(VISUAL_COMPLEXITY.get(t, 0.30) for t in self.img_type)



class MmmuPruner:
    """
    Single-model pruner for MMMU benchmark compression.

    Because MMMU ships with only one model (glm-4.5v-fp8), cross-model
    discrimination is unavailable. This pruner uses two proxy signals:

    1. topic_difficulty ('Easy'/'Medium'/'Hard') — directly indicates
       how challenging the sample is; Hard samples are most discriminating.

    2. img_type visual complexity — samples requiring precise visual parsing
       (Chemical Structures, Technical Blueprints) are more sensitive to
       encoder quality than samples requiring only coarse recognition
       (Photographs, Paintings).

    Subject diversity guarantee ensures the pruned set covers all 30 MMMU
    subjects, preventing the probe from being dominated by any single domain.

    Args:
        prune_ratio: Fraction of samples to keep (default 0.2 for MMMU)
        min_per_subject: Minimum samples to keep per MMMU subject
        difficulty_weights: Budget allocation per difficulty bin
    """

    def __init__(
        self,
        prune_ratio: float = 0.2,
        min_per_subject: int = 1,
        difficulty_weights: Optional[Dict[str, float]] = None,
    ):
        if not 0 < prune_ratio <= 1.0:
            raise ValueError(f'prune_ratio must be between 0 and 1, got {prune_ratio}')

        self.prune_ratio = prune_ratio
        self.min_per_subject = min_per_subject
        self.difficulty_weights = difficulty_weights or DEFAULT_DIFFICULTY_WEIGHTS

        if abs(sum(self.difficulty_weights.values()) - 1.0) > 1e-6:
            raise ValueError('difficulty_weights must sum to 1.0')

    def select_samples(self, samples: List[MmmuSample]) -> List[int]:
        """
        Select sample indices for the pruned set.

        Returns:
            Sorted list of selected sample indices.
        """
        if not samples:
            return []

        n_subjects = len(set(s.subject for s in samples))
        total_budget = max(
            n_subjects * self.min_per_subject,
            int(len(samples) * self.prune_ratio),
        )

        selected_indices: set = set()

        # Step 1 — Subject diversity: select the best sample per subject.
        # Prefer: incorrect > correct; within that, higher visual complexity.
        by_subject: Dict[str, List[MmmuSample]] = defaultdict(list)
        for s in samples:
            by_subject[s.subject].append(s)

        for subject_samples in by_subject.values():
            best = max(subject_samples, key=lambda s: s.visual_complexity)
            selected_indices.add(best.index)

        # Step 2 — Fill remaining budget by difficulty × visual complexity.
        remaining = total_budget - len(selected_indices)
        if remaining <= 0:
            return sorted(selected_indices)

        unselected = [s for s in samples if s.index not in selected_indices]
        by_difficulty: Dict[str, List[MmmuSample]] = defaultdict(list)
        for s in unselected:
            by_difficulty[s.topic_difficulty].append(s)

        for difficulty, weight in self.difficulty_weights.items():
            bin_budget = max(0, int(remaining * weight))
            bin_samples = by_difficulty.get(difficulty, [])
            bin_samples.sort(key=lambda s: s.visual_complexity, reverse=True)
            for s in bin_samples[:bin_budget]:
                selected_indices.add(s.index)

        return sorted(selected_indices)

    def get_pruning_stats(
        self,
        all_samples: List[MmmuSample],
        selected_indices: List[int],
    ) -> Dict:
        """Return statistics about the pruning for reporting."""
        selected_set = set(selected_indices)
        selected = [s for s in all_samples if s.index in selected_set]

        full_acc = (
            sum(s.score for s in all_samples) / len(all_samples)
            if all_samples else 0.0
        )
        pruned_acc = (
            sum(s.score for s in selected) / len(selected)
            if selected else 0.0
        )

        return {
            'total_samples': len(all_samples),
            'selected_samples': len(selected_indices),
            'actual_prune_ratio': len(selected_indices) / len(all_samples) if all_samples else 0,
            'full_accuracy': round(full_acc, 4),
            'pruned_accuracy': round(pruned_acc, 4),
            'difficulty_distribution': dict(Counter(s.topic_difficulty for s in selected)),
            'subject_coverage': len(set(s.subject for s in selected)),
            'total_subjects': len(set(s.subject for s in all_samples)),
            'top_img_types': Counter(
                t for s in selected for t in s.img_type
            ).most_common(5),
        }


def load_mmmu_samples(
    predictions_dir: str,
    reviews_dir: str,
) -> List[MmmuSample]:
    """
    Load MMMU samples by joining predictions + reviews JSONL files.

    File pattern: mmmu_{Subject}.jsonl
    Subject is extracted from the filename (e.g. mmmu_Accounting.jsonl -> 'Accounting').

    Args:
        predictions_dir: Path to MMMU predictions directory
        reviews_dir: Path to MMMU reviews directory

    Returns:
        List of MmmuSample objects with metadata and scores
    """
    predictions_path = Path(predictions_dir)
    reviews_path = Path(reviews_dir)

    review_files = sorted(reviews_path.glob('mmmu_*.jsonl'))
    if not review_files:
        raise FileNotFoundError(
            f'No MMMU review files found in {reviews_dir}'
        )

    samples = []
    global_idx = 0

    for review_file in review_files:
        subject = review_file.stem.replace('mmmu_', '')

        # Load per-sample scores from reviews
        scores: Dict[int, float] = {}
        with open(review_file) as f:
            for line in f:
                row = json.loads(line.strip())
                idx = row['index']
                score = float(row['sample_score']['score']['value'].get('acc', 0.0))
                scores[idx] = score

        # Load metadata from predictions
        pred_file = predictions_path / review_file.name
        if not pred_file.exists():
            continue

        with open(pred_file) as f:
            for line in f:
                row = json.loads(line.strip())
                idx = row['index']
                if idx not in scores:
                    continue

                meta = row.get('metadata', {})
                raw_img_type = meta.get('img_type', [])
                if isinstance(raw_img_type, str):
                    raw_img_type = ast.literal_eval(raw_img_type)

                samples.append(MmmuSample(
                    index=global_idx,
                    subject=subject,
                    subfield=meta.get('subfield', ''),
                    topic_difficulty=meta.get('topic_difficulty', 'Medium'),
                    img_type=raw_img_type,
                    score=scores[idx],
                ))
                global_idx += 1

    return samples
