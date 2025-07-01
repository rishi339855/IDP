import numpy as np

def is_yawning(landmarks):
    top = np.mean([landmarks[13], landmarks[14]], axis=0)
    bottom = np.mean([landmarks[17], landmarks[18]], axis=0)
    distance = np.linalg.norm(top - bottom)
    return distance > 25  # Adjust threshold as per resolution
