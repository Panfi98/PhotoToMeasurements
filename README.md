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
