"""
DASS Score Label Loader for EMOTHAW Dataset.

Reads DASS_scores.xls and maps user IDs to emotional state labels
(depression, anxiety, stress) using standard DASS-21 severity thresholds.
"""
import os
import xlrd
import numpy as np


def load_dass_scores(dass_path: str) -> dict:
    """
    Load raw DASS scores from the Excel file.

    Args:
        dass_path: Path to DASS_scores.xls

    Returns:
        Dict mapping user_id (int) → {
            'depression': float, 'anxiety': float, 'stress': float,
            'collection': str
        }
    """
    wb = xlrd.open_workbook(dass_path)
    sh = wb.sheet_by_index(0)

    scores = {}
    for row_idx in range(1, sh.nrows):
        row = sh.row_values(row_idx)
        try:
            subject = int(row[0])
            depression = float(row[1])
            anxiety = float(row[2])
            stress = float(row[3])
            collection = str(row[6]).strip()  # 'Collection1' or 'Collection2'

            # Map subject number to user_id
            # In EMOTHAW, subject number directly maps to user folder number
            file_num = int(row[5]) if row[5] != "" else subject
            user_id = file_num

            scores[user_id] = {
                "depression": depression,
                "anxiety": anxiety,
                "stress": stress,
                "collection": collection,
            }
        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping row {row_idx}: {e}")

    return scores


def score_to_severity(score: float, emotion: str, thresholds: dict) -> int:
    """
    Convert a raw DASS score to a severity level index.

    Args:
        score: Raw DASS score.
        emotion: One of 'depression', 'anxiety', 'stress'.
        thresholds: DASS threshold dict from config.

    Returns:
        Severity index: 0=Normal, 1=Mild, 2=Moderate, 3=Severe, 4=ExSevere
    """
    cuts = thresholds[emotion]
    if score < cuts[1]:
        return 0  # Normal
    elif score < cuts[2]:
        return 1  # Mild
    elif score < cuts[3]:
        return 2  # Moderate
    elif score < cuts[4]:
        return 3  # Severe
    else:
        return 4  # Extremely Severe


def score_to_binary(score: float, emotion: str, binary_thresholds: dict) -> int:
    """
    Convert a raw DASS score to a binary label.
    0 = Low (Normal + Mild), 1 = High (Moderate + Severe + ExSevere)
    """
    return 0 if score < binary_thresholds[emotion] else 1


def load_labels(dass_path: str = None,
                target_emotion: str = "anxiety",
                use_binary: bool = True) -> dict:
    """
    Load labels for all users.

    Args:
        dass_path: Path to DASS_scores.xls. If None, uses config default.
        target_emotion: Which emotion to classify ('depression', 'anxiety', 'stress').
        use_binary: If True, binary labels (Low/High). If False, 5-class severity.

    Returns:
        Dict mapping user_id → int label.
    """
    if dass_path is None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import DASS_SCORES_PATH
        dass_path = DASS_SCORES_PATH

    from config import DASS_THRESHOLDS, BINARY_THRESHOLD

    scores = load_dass_scores(dass_path)
    labels = {}

    for user_id, user_scores in scores.items():
        raw_score = user_scores[target_emotion]
        if use_binary:
            labels[user_id] = score_to_binary(raw_score, target_emotion, BINARY_THRESHOLD)
        else:
            labels[user_id] = score_to_severity(raw_score, target_emotion, DASS_THRESHOLDS)

    return labels


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DASS_SCORES_PATH, TARGET_EMOTION, USE_BINARY, CLASS_NAMES

    labels = load_labels(DASS_SCORES_PATH, TARGET_EMOTION, USE_BINARY)
    print(f"Loaded labels for {len(labels)} users")
    print(f"Target emotion: {TARGET_EMOTION}, Binary: {USE_BINARY}")
    print(f"Class distribution:")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        count = sum(1 for v in labels.values() if v == cls_idx)
        print(f"  {cls_name}: {count}")
    print(f"\nSample labels: {dict(list(labels.items())[:5])}")
