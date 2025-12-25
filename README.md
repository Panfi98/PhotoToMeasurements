# Photo to measurements

Proof of concept for estimating human body measurements (body, chest, hips, inseam, sleeve length, etc) can be estimated from 
 - 1 front photo
 - 1 side photo
 - Declared body height

### Core requirenments
- Fail fast on bad images (Image quality and pose correctness are enforced before any measurement attemps)
- Geometry first, ML second (Measurements are derived primarily from geometry and body, not black box regressions)
- Confidence-aware outputs (Every estimated measurements includes confiedence score based on sanity checks)
- Debuggable by design (All intermidiate steps are visualized: keypoints, masks, measurement lines)


### Machine Learning 
This project will use MediaPipe from google - a set of libraries and tools that provide inference machine learning capabilities to process images. 

Under the hood flow example:

1. First the image is converted into a tensor - a multi-dimensional array of number. Since the neural networks only understand numbers. Neural networks expect input in a very specific numeric structur. 

2. The pose neural network runs forward porpagation

3. The model predicts:
    - Body keypoints
    - Confidence scores
4. MediaPipe post-processes results:
    - Normalizes coordinates
    - Filters noise
    - Packs results into pose_landmarks