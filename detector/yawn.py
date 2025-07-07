import numpy as np

def is_yawning(landmarks, debug=False):
    """
    Robust yawn detection using MediaPipe face mesh landmarks.
    Uses the ratio of mouth opening to face size for better accuracy.
    If debug=True, returns (is_yawn, mouth_ratio, mouth_distance, face_width)
    """
    try:
        # Upper lip landmarks (average)
        upper_lip_indices = [13, 14, 15, 16]
        upper_lip = np.mean([landmarks[i] for i in upper_lip_indices], axis=0)
        # Lower lip landmarks (average)
        lower_lip_indices = [17, 18, 19, 20]
        lower_lip = np.mean([landmarks[i] for i in lower_lip_indices], axis=0)
        # Mouth opening
        mouth_distance = np.linalg.norm(upper_lip - lower_lip)
        # Face width for normalization (cheek to cheek)
        face_width = np.linalg.norm(landmarks[10] - landmarks[9])
        if face_width > 0:
            mouth_ratio = mouth_distance / face_width
            is_yawn = mouth_ratio > 0.30
        else:
            mouth_ratio = 0
            is_yawn = mouth_distance > 50
        if debug:
            return is_yawn, mouth_ratio, mouth_distance, face_width
        return is_yawn
    except (IndexError, ValueError, TypeError):
        if debug:
            return False, 0, 0, 0
        return False