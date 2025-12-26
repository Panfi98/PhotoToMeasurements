import os
import json

import cv2
import numpy as np
import mediapipe as mp
from typing import Any, Optional

drawing_utils = mp.solutions.drawing_utils
pose_solution = mp.solutions.pose


def ensure_directory_exists(directory_path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(directory_path, exist_ok=True)


def draw_pose(
    image_bgr: np.ndarray,
    pose_result: Optional[Any]
) -> np.ndarray:
    """
    Draw pose landmarks on an image.

    Args:
        image_bgr: Input image in BGR format.
        pose_result: MediaPipe pose detection result.

    Returns:
        Image with pose landmarks drawn.
    """
    image = image_bgr.copy()

    if pose_result and pose_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image,
            pose_result.pose_landmarks,
            pose_solution.POSE_CONNECTIONS,
        )
    return image


def overlay_mask(
    image_bgr: np.ndarray,
    mask_prob: np.ndarray,
    alpha: float = 0.45,
    overlay_color_bgr: tuple = (0, 255, 0),
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Overlay a semi-transparent color on image regions where mask_prob > threshold.

    Args:
        image_bgr: Input image in BGR format.
        mask_prob: Probability mask array.
        alpha: Transparency level for the overlay (0-1).
        overlay_color_bgr: Color to overlay in BGR format.
        threshold: Probability threshold for applying the mask.

    Returns:
        Image with the mask overlay applied.
    """
    output_image = image_bgr.copy()
    binary_mask = mask_prob > threshold
    overlay_color = np.array(overlay_color_bgr, dtype=np.float32)

    output_image[binary_mask] = (
        output_image[binary_mask].astype(np.float32) * (1 - alpha)
        + overlay_color * alpha
    ).astype(np.uint8)

    return output_image
