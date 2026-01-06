# Preprocessing

This folder contains scripts and notebooks for preprocessing raw maize plant images before analysis. The preprocessing pipeline prepares images for object detection, segmentation, and classification tasks.

## Overview

Raw images from side-view cameras require several preprocessing steps to:
- Organize images by scanning session (morning/evening)
- Normalize lighting and color variations
- Extract individual plant patches from group images
- Remove background and adjacent plant interference

## Files

### Python Scripts

#### `Session_divide.py`
**Purpose**: Organizes daily images into separate scanning sessions

**Functionality**:
- Scans through trial directories to find all imaging dates
- Extracts timestamp information from image filenames
- Separates images into Session1 (morning) and Session2 (evening)
- Creates organized directory structure: `Sessions/{date}/Session{1,2}/`

**Key Parameters**:
- `src_path`: Source directory containing raw trial images
- `trial_num`: Trial identifier (e.g., '005')

**Usage**:
```bash
python Session_divide.py
```

**Output**:
```
Sessions/
├── y20m09d27/
│   ├── Session1/  # Morning scan
│   └── Session2/  # Evening scan
├── y20m09d28/
│   ├── Session1/
│   └── Session2/
...
```

---

#### `ImageNormalization.py`
**Purpose**: Normalizes image intensities across all channels using linear scaling

**Methodology**:
- Applies channel-wise normalization: `x' = 255 × (x - x_min) / (x_max - x_min)`
- Calculates min/max values per session to maintain temporal consistency
- Processes Blue, Green, and NIR (Near-Infrared) channels independently

**Normalization Process**:
1. Load all images in a session
2. Calculate mean of minimum pixel values per channel
3. Calculate mean of maximum pixel values per channel
4. Apply linear scaling to normalize to [0, 255] range
5. Save normalized images

**Key Parameters**:
- `trial_num`: Trial identifier
- `src`: Session directory path
- `toDirectory`: Output path for normalized images

**Usage**:
```bash
python ImageNormalization.py
```

**Technical Details**:
- Preserves relative intensity relationships within each session
- Handles RGB and NIR channels separately
- Creates NormalizedImages directory structure mirroring input

---

#### `Session_divide.py`
**Purpose**: Organizes images by scanning timestamp

**Algorithm**:
1. Parse timestamps from image filenames (format: `hr_min_sec`)
2. Identify unique timestamps per day
3. Group images by timestamp
4. Copy to session-specific subdirectories

**Filename Convention**:
Images are expected to have timestamps embedded at character positions 15:23

---

#### `remove_images.py`
**Purpose**: Utility script for removing unwanted or corrupted images

**Use Cases**:
- Remove corrupted files (e.g., Thumbs.db)
- Filter out test images
- Clean up failed processing outputs

---

#### `count_files_in_trial.py`
**Purpose**: Data validation utility to count images per trial/session

**Functionality**:
- Counts files across trial directory structure
- Verifies consistent image capture across days
- Identifies missing data points

---

### Jupyter Notebooks

#### `CutImages.ipynb`
**Purpose**: Semi-automatic extraction of individual plant patches from group images

**Process**:
1. Display raw image showing all 4 plants
2. User defines bounding boxes for each plant
3. Script extracts and saves individual plant images
4. Creates separate directories for Plant1, Plant2, Plant3, Plant4

**Interactive Steps**:
- Load image
- Click to define patch boundaries
- Verify extraction
- Process entire trial automatically using defined parameters

**Tutorial Resources**:
- Basic tutorial: [Zoom recording](https://ncsu.zoom.us/rec/share/rGn0sQA23MXq3vZwPn7ginRRbDXW-M3RAzYWXm5N5fkFU3-UyjSomOvfFtzsm75X.yWi8Harb2fWXiSJC) (Password: *zUSgX3%)
- Lab session: [Zoom recording](https://ncsu.zoom.us/rec/share/MafBagPOYAyZXNsXN6CDk1vs86FZ3ugIqs0JgcmX4AHx1NinDeFs5auuMCyN_r-3.C8O0luySiKnvojEz) (Password: H#m?M588)

**Output Structure**:
```
Patches/
├── Plant1/
│   ├── y20m09d27/
│   │   └── Scans/
│   │       ├── 1/  # Morning scan patches
│   │       └── 2/  # Evening scan patches
├── Plant2/  # Drought stressed
├── Plant3/
└── Plant4/  # Drought stressed
```

---

#### `Blackout.ipynb`
**Purpose**: Iteratively removes background and adjacent plant interference

**Technique**:
- Adds black rectangular blocks to mask unwanted regions
- Isolates target plant from neighboring plants
- Preserves plant-of-interest while eliminating confounding factors

**Interactive Workflow**:
1. Display plant patch
2. User defines blackout regions (coordinates)
3. Apply black masks
4. Verify masking quality
5. Batch process all images with saved parameters

**Tutorial**:
- Step-by-step guide: [Zoom recording](https://ncsu.zoom.us/rec/share/OTcNoli7DiLgFxhHaAdHyHQbKqc4FtfDew9caCC-fBKBQjyMcQXkYiYO3mzKG-UA.1jftVTne5G3tjC1L) (Password: i$Par6Zt)

**Application**:
Critical for preventing model confusion when multiple plants appear in frame

---

## Preprocessing Pipeline Order

Follow this sequence for complete preprocessing:

```
1. Session_divide.py          → Organize by time of day
2. ImageNormalization.py      → Standardize intensities
3. CutImages.ipynb           → Extract individual plants
4. Blackout.ipynb            → Remove interference
```

## Dependencies

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
from sklearn.feature_extraction import image
import pandas as pd
import glob
```

## Configuration

### Path Configuration
Update these paths in each script:
```python
# Session_divide.py
src_path = 'D:/Plant Water Stress/Data/Trial_005/Trial005/'

# ImageNormalization.py
root = 'D:/Plant Water Stress/Data/Trial_005/Trial005'
src = "D:/Plant Water Stress/Data/Trial_005/Sessions/"
toDirectory = "D:/Plant Water Stress/Data/Trial_005/NormalizedImages/"
```

### Trial Selection
```python
trial_num = '005'  # Change to your trial number
```

## Input Requirements

### Image Format
- File type: JPEG or similar formats
- Color space: RGB or RGB + NIR
- Naming convention: Must include timestamp in consistent format
- Expected dimensions: High-resolution plant images

### Directory Structure
Raw data should be organized:
```
Trial_XXX/
├── TrialXXX/
│   ├── y20m09d27/
│   │   ├── image_timestamp1.jpg
│   │   ├── image_timestamp2.jpg
│   │   └── ...
```

## Output

### Normalized Images
- Format: JPEG
- Channels: Normalized to [0, 255]
- Preserves: Original image dimensions and color depth

### Patches
- Individual plant images
- Consistent framing across days
- Ready for model input

### Logs
- Session counts
- File processing confirmations
- Error messages for corrupted files

## Quality Control

### Verification Steps
1. Check image counts per session (should be consistent)
2. Visually inspect normalized images for artifacts
3. Verify patch boundaries align with plant positions
4. Confirm blackout masks don't obscure plant-of-interest

### Common Issues

**Issue**: Inconsistent session counts
- **Solution**: Check for missing image files or timestamp parsing errors

**Issue**: Over/under-normalization
- **Solution**: Verify min/max calculations are per-session, not per-image

**Issue**: Poor patch extraction
- **Solution**: Adjust bounding box coordinates in CutImages.ipynb

**Issue**: Insufficient blackout masking
- **Solution**: Iteratively add more blackout regions in Blackout.ipynb

## Performance Considerations

- Normalization processes entire sessions to maintain consistency
- Patch extraction is semi-automatic but requires initial user input
- Blackout masking may need adjustment per plant/trial
- Processing time scales linearly with image count

## Tips for Best Results

1. **Consistent Lighting**: Normalization works best with relatively consistent lighting
2. **Careful Patch Boundaries**: Take time to accurately define plant boundaries
3. **Complete Blackout**: Ensure no adjacent plant pixels remain visible
4. **Validation**: Always visually check outputs before proceeding to next stage

## Troubleshooting

### Script Fails to Find Images
- Verify path separators (use `os.path.join()`)
- Check trial_num matches directory name
- Ensure timestamp parsing matches your filename format

### Normalization Creates Artifacts
- Check for corrupted input images
- Verify channel ordering (BGR vs RGB)
- Ensure sufficient bit depth in input images

### Patches Miss Plant Regions
- Review bounding box coordinates
- Check for plant movement between sessions
- Consider plant growth over trial duration

## Next Steps

After preprocessing, images are ready for:
- Object detection (Labelbox_Detectron2)
- Segmentation tasks
- Classification with Vision Transformers
- Histogram analysis

## Related Documentation

- Main README: `../README.md`
- Object Detection: `../Labelbox_Detectron2/README.md`
- Analysis: `../Histograms/README.md`
