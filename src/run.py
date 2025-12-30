import os
import json
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model paths and URLs from environment
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "pose_landmarker.task")
SEGMENTER_MODEL_PATH = os.getenv("SEGMENTER_MODEL_PATH", "selfie_segmenter.tflite")
POSE_MODEL_URL = os.getenv("POSE_MODEL_URL", "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
SEGMENTER_MODEL_URL = os.getenv("SEGMENTER_MODEL_URL", "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite")


def download_model_if_needed(model_path: str, url: str) -> None:
    """Download a model file if it doesn't exist locally."""
    if not os.path.exists(model_path):
        print(f"Downloading {model_path}...")
        urllib.request.urlretrieve(url, model_path)
        print(f"Downloaded {model_path}")


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
        for landmarks in pose_result.pose_landmarks:
            # Draw landmarks as circles
            for landmark in landmarks:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            # Draw connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 7),  # Face
                (0, 4), (4, 5), (5, 6), (6, 8),  # Face
                (9, 10),  # Mouth
                (11, 12),  # Shoulders
                (11, 13), (13, 15),  # Left arm
                (12, 14), (14, 16),  # Right arm
                (11, 23), (12, 24),  # Torso
                (23, 24),  # Hips
                (23, 25), (25, 27),  # Left leg
                (24, 26), (26, 28),  # Right leg
            ]
            for start_idx, end_idx in connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]
                    start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
                    end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)

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
        mask_prob: Probability mask array (can be 2D or with extra dimension).
        alpha: Transparency level for the overlay (0-1).
        overlay_color_bgr: Color to overlay in BGR format.
        threshold: Probability threshold for applying the mask.

    Returns:
        Image with the mask overlay applied.
    """
    output_image = image_bgr.copy()

    # Ensure mask is 2D
    if mask_prob.ndim > 2:
        mask_prob = mask_prob.squeeze()

    # Resize mask to match image if needed
    if mask_prob.shape[:2] != image_bgr.shape[:2]:
        mask_prob = cv2.resize(mask_prob, (image_bgr.shape[1], image_bgr.shape[0]))

    binary_mask = mask_prob > threshold

    # Create colored overlay
    overlay = np.zeros_like(output_image)
    overlay[:] = overlay_color_bgr

    # Apply overlay where mask is True
    output_image[binary_mask] = (
        output_image[binary_mask].astype(np.float32) * (1 - alpha)
        + overlay[binary_mask].astype(np.float32) * alpha
    ).astype(np.uint8)

    return output_image

def detect_pose(img_rgb: np.ndarray) -> Any:
    """
    Detect pose landmarks in an RGB image.

    Args:
        img_rgb: Input image in RGB format.

    Returns:
        MediaPipe pose detection result.
    """
    download_model_if_needed(POSE_MODEL_PATH, POSE_MODEL_URL)

    base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    return detector.detect(mp_image)


def segment_person(img_rgb: np.ndarray) -> np.ndarray:
    """
    Segment a person from the background.

    Args:
        img_rgb: Input image in RGB format.

    Returns:
        Segmentation mask as a probability array.
    """
    download_model_if_needed(SEGMENTER_MODEL_PATH, SEGMENTER_MODEL_URL)

    base_options = python.BaseOptions(model_asset_path=SEGMENTER_MODEL_PATH)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True,
    )
    segmenter = vision.ImageSegmenter.create_from_options(options)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = segmenter.segment(mp_image)

    # Get category mask and convert to probability-like format
    category_mask = result.category_mask.numpy_view()
    return category_mask.astype(np.float32)


def save_mask_as_png(mask: np.ndarray, output_path: str) -> None:
    """Save a probability/category mask as an 8-bit PNG image."""
    # Ensure mask is 2D
    if mask.ndim > 2:
        mask = mask.squeeze()

    # Normalize and convert to uint8
    mask_normalized = mask.astype(np.float32)
    if mask_normalized.max() > 1:
        mask_normalized = mask_normalized / mask_normalized.max()
    mask_uint8 = (mask_normalized * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_uint8)


def find_front_image(subject_dir: str) -> str:
    """
    Find the front-facing image in a subject directory.

    Args:
        subject_dir: Path to the subject directory.

    Returns:
        Path to the front image.

    Raises:
        FileNotFoundError: If no front image is found.
    """
    # Possible names for the front image (case-insensitive)
    possible_names = ["front.jpg", "front.jpeg", "front.png", "front_side.jpg", "front_side.jpeg", "front_side.png"]

    for filename in os.listdir(subject_dir):
        if filename.lower() in possible_names or filename.lower().replace(".jpg", ".jpeg") in possible_names:
            return os.path.join(subject_dir, filename)

    # Also check for .JPG (uppercase)
    for filename in os.listdir(subject_dir):
        lower_name = filename.lower()
        if "front" in lower_name and lower_name.endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(subject_dir, filename)

    raise FileNotFoundError(f"No front image found in {subject_dir}")


def main(subject_dir: str = "input/subject_01") -> None:
    """
    Process a subject's front image for pose detection and segmentation.

    Args:
        subject_dir: Path to the subject directory containing front.jpg.

    Raises:
        FileNotFoundError: If front.jpg is missing.
        ValueError: If the image cannot be read by OpenCV.
    """
    front_path = find_front_image(subject_dir)
    print(f"Processing: {front_path}")

    out_dir = os.path.join("output", os.path.basename(subject_dir))
    ensure_directory_exists(out_dir)

    img_bgr = cv2.imread(front_path)
    if img_bgr is None:
        raise ValueError(f"OpenCV could not read the image: {front_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Pose detection
    pose_result = detect_pose(img_rgb)
    pose_overlay = draw_pose(img_bgr, pose_result)
    cv2.imwrite(os.path.join(out_dir, "front_pose.jpg"), pose_overlay)

    # Segmentation
    mask = segment_person(img_rgb)
    save_mask_as_png(mask, os.path.join(out_dir, "front_mask.png"))

    mask_overlay = overlay_mask(img_bgr, mask)
    cv2.imwrite(os.path.join(out_dir, "front_mask_overlay.jpg"), mask_overlay)

    # Quality metrics
    quality = {
        "pose_has_landmarks": len(pose_result.pose_landmarks) > 0 if pose_result.pose_landmarks else False,
        "segmentation_mean": float(np.mean(mask)),
    }
    with open(os.path.join(out_dir, "quality.json"), "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    print(f"Saved outputs to: {out_dir}")

if __name__ == "__main__":
    main()