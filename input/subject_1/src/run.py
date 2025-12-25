import os, json
import cv2
import numpy as np
import mediapipe as mp
from typing import Any, Optional

drawing_utils = mp.solutions.drawing_utils
pose_solution = mp.solution.pose
 
def ensure_directory_exists( directory_path: str ) -> None:
    os.makedirs(directory_path, exist_ok = True)


def draw_pose(
        image_bgr: np.ndarray,
        pose_result: Optional[any]
        ) -> np.ndarray:
    image = image_bgr.copy()
    
    if pose_result and pose_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image,
            pose_result.pose_landmarks,
            pose_solution.POSE_CONNECTIONS,
        )
    return image

