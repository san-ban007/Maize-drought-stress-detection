# Preprocessing

This folder contains scripts and notebooks for preprocessing raw maize plant images before analysis. The preprocessing pipeline prepares images for object detection, segmentation, and classification tasks.

## Overview

Raw images collected from raspberry pi cameras require several preprocessing steps to:
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
- Remove corrupted files
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

