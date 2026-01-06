# Labelbox_Detectron2

Object detection and segmentation pipeline using Detectron2 with Labelbox annotation integration for identifying and extracting maize plant components 

## Overview

This module implements a complete workflow for:
1. Training object detection models using labeled data from Labelbox
2. Generating bounding box predictions across entire datasets
3. Extracting and segmenting specific plant components based on predictions

The system identifies two primary plant components:
- **YEL (Youngest Expanding Leaf)**: Class 0 - The youngest, most stress-sensitive leaf
- **Stem**: Class 1 - The main plant stem structure

## Architecture

The pipeline uses:
- **Framework**: Detectron2 (Facebook AI Research)
- **Base Model**: Faster R-CNN with ResNet-50-FPN backbone
- **Pre-training**: COCO dataset
- **Fine-tuning**: Custom maize plant annotations from Labelbox
- **Annotation Tool**: Labelbox for collaborative labeling

## Files

### `Detectron2_train.ipynb`
**Purpose**: Train a Detectron2 object detection model using Labelbox annotations

**Key Features**:
- Connects to Labelbox API to fetch annotations
- Converts Labelbox format to COCO format
- Registers custom datasets with Detectron2
- Fine-tunes Faster R-CNN on maize plant data
- Evaluates model performance with COCO metrics

**Training Configuration**:
```python
model_zoo_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # YEL and Stem
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Confidence threshold
```

**Workflow**:
1. Initialize Labelbox client with API key
2. Fetch project annotations
3. Serialize to COCO format
4. Split into train/test sets (typically 90/10)
5. Register datasets with Detectron2 metadata
6. Configure training parameters
7. Train model
8. Evaluate on test set
9. Save model weights

**Output**:
- `model_yelstem_XXX.pth`: Trained model weights
- Training logs and loss curves
- Evaluation metrics (mAP, precision, recall)

---

### `Bbox_Generate.py`
**Purpose**: Generate bounding box predictions across all trial images

**Functionality**:
- Loads trained Detectron2 model
- Processes images systematically by Plant and Date
- Runs inference to detect YEL and Stem regions
- Saves visualization with bounding boxes
- Exports predictions to text file format

**Processing Loop**:
```python
For each Plant (1-4):
    For each Date in trial:
        For each image in scan:
            1. Load image
            2. Run detection model
            3. Extract predictions (classes, bounding boxes)
            4. Visualize detections
            5. Save results
```

**Key Configuration**:
```python
trial_num = '008'
root = '/path/to/Trial008/Patches'
Dates = ['y20m12d14', 'y20m12d15', ...]  # Trial dates
Scan = '1'  # 1=morning, 2=evening
Plants = ['Plant1', 'Plant2', 'Plant3', 'Plant4']
```

**Prediction Format**:
```
image_path,class:box_coords,class:box_coords,...
```
Example:
```
/path/Plant1/y20m12d14/Scans/1/img0001.jpg,0:[x1 y1 x2 y2],1:[x1 y1 x2 y2]
```

**Output Structure**:
```
Plant1/
├── y20m12d14/
│   └── Scans/
│       └── 1/
│           ├── BBys_T008/           # Visualizations with boxes
│           │   ├── img0001.jpg
│           │   └── ...
│           └── PREDys_NEW/          # Prediction text files
│               └── predictions.txt
```

---

### `Predictions_Extract.py`
**Purpose**: Extract and segment YEL and Stem regions from predicted bounding boxes

**Two-Stage Processing**:

**Stage 1: YEL Extraction**
- Parses prediction text files
- Identifies Class 0 (YEL) bounding boxes
- Masks everything outside bounding box with white pixels
- Applies HSV color segmentation to isolate green tissue
- Saves segmented YEL images

**Stage 2: Stem Extraction**
- Same process but for Class 1 (Stem)
- Extracts stem regions
- Applies color segmentation
- Saves segmented stem images

**Color Segmentation Algorithm**:
```python
# Convert to HSV color space
# Define masks:
lower_mask = hue > 0.5      # Select yellow-green range
upper_mask = hue < 0.9      # Exclude red/purple
saturation_mask = saturation > 0.5  # Remove white/gray

# Apply combined mask
mask = upper_mask * lower_mask * saturation_mask
segmented_image = original_image * mask
```

**Processing Logic**:
```python
For each prediction line:
    If no detections:
        Fill image with white (255,255,255)
    If Class 0 found (YEL):
        Crop to bounding box
        Apply color segmentation
        Save to YELys_NEW/
    If Class 1 found (Stem):
        Crop to bounding box
        Apply color segmentation
        Save to STEMys_NEW/
```

**Key Functions**:

`image_processing(image, top, left, right, bottom)`:
- Masks regions outside bounding box with white

`color_segmentation(image)`:
- Converts to HSV
- Applies hue and saturation thresholds
- Returns masked RGB image

`all_white(image)`:
- Creates blank white image (no detection case)

**Output**:
```
Plant1/
├── y20m09d27/
│   └── Scans/
│       └── 1/
│           ├── YELys_NEW/    # Segmented YEL images
│           │   ├── 0001.jpg
│           │   └── ...
│           └── STEMys_NEW/   # Segmented Stem images
│               ├── 0001.jpg
│               └── ...
```

---

## Labelbox Integration

### Setup
1. Create Labelbox account at [labelbox.com](https://labelbox.com)
2. Generate API key from settings
3. Create new project for maize annotation
4. Define ontology (YEL and Stem classes)

### Annotation Guidelines

**YEL**:
- Draw tight bounding box around youngest leaf
- Include entire leaf blade but exclude stem
- One box per image (most prominent YEL)

**Stem**:
- Draw box around visible main stem
- Exclude leaves and branches
- Include full vertical extent visible in frame

## Model Training

### Hyperparameters
```python
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
```

### Training Tips
1. **Data Augmentation**: Detectron2 applies random flips automatically
2. **Learning Rate**: Start with 0.00025, reduce if loss plateaus
3. **Iterations**: Monitor validation loss, stop when converging
4. **Batch Size**: Adjust based on GPU memory availability

### Evaluation Metrics
- **mAP (mean Average Precision)**: Overall detection quality
- **AP@IoU=0.50**: Precision at 50% overlap threshold
- **AP@IoU=0.75**: Precision at 75% overlap threshold
- **Per-class AP**: Individual performance for YEL and Stem

## Dependencies

```bash
pip install torch torchvision
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install labelbox
pip install opencv-python
pip install scikit-image
pip install shapely
pip install tqdm
```

## Configuration

### API Keys
Replace placeholder API keys in scripts:
```python
# In Bbox_Generate.py
API_KEY = 'your_labelbox_api_key_here'
```

### Paths
Update all hardcoded paths to match your system:
```python
# Bbox_Generate.py
root = '/mnt/research-projects/.../Trial008/Patches'

# Predictions_Extract.py
root = '/mnt/research-projects/.../Trial005/Patches'
```

### Trial Configuration
```python
trial_num = '008'  # Your trial number
Dates = ['y20m12d14', 'y20m12d15', ...]  # Specific to your trial
Scan = '1'  # 1=morning, 2=evening
```

## Usage

### Step 1: Train Model
```bash
# Open and run notebook
jupyter notebook Detectron2_train.ipynb

# Follow cells sequentially:
# 1. Connect to Labelbox
# 2. Download annotations
# 3. Register dataset
# 4. Train model
# 5. Evaluate results
```

### Step 2: Generate Predictions
```bash
# Edit paths and trial configuration
python Bbox_Generate.py

# Monitor progress
# - Processing messages per image
# - Bounding box visualizations created
# - predictions.txt generated per scan
```

### Step 3: Extract Segments
```bash
# Edit paths to match your trial
python Predictions_Extract.py

# Processes twice (YEL then Stem)
# Creates segmented images in respective folders
```

## Output Interpretation

### Prediction Text Format
```
filepath,class:coordinates,class:coordinates
```

Where:
- `class`: 0=YEL, 1=Stem
- `coordinates`: [x_min, y_min, x_max, y_max] in pixels

### Visualization Images
- Original image with colored bounding boxes overlaid
- YEL boxes typically in one color
- Stem boxes in another color
- Confidence scores displayed (if enabled)

### Segmented Images
- White background (255,255,255)
- Plant tissue in original RGB colors
- Clean separation of YEL or Stem
- Ready for downstream analysis (histograms, classification)

## Quality Assurance

### Visual Inspection
1. Check bounding box visualizations for accuracy
2. Verify segmentation removes background properly
3. Ensure color thresholds don't exclude valid tissue
4. Look for missed detections (empty predictions)

### Common Issues

**Issue**: Low confidence scores
- **Solution**: Collect more training annotations
- **Solution**: Adjust `cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST`

**Issue**: Missing detections
- **Solution**: Check if YEL/Stem visible in original image
- **Solution**: Review color segmentation thresholds
- **Solution**: Retrain with more diverse examples

**Issue**: Over-segmentation
- **Solution**: Increase confidence threshold
- **Solution**: Apply non-maximum suppression

**Issue**: Color segmentation too aggressive
- **Solution**: Adjust HSV thresholds in `color_segmentation()`
- **Solution**: Test on range of lighting conditions

## Performance Optimization

### GPU Utilization
```python
# Check GPU availability
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Increase batch size if memory allows
cfg.SOLVER.IMS_PER_BATCH = 4  # Default is 2
```

### Parallel Processing
The current implementation processes images sequentially. For large datasets, consider:
- Multi-GPU training with `DistributedDataParallel`
- Batch inference for predictions
- Multi-threaded image I/O

## Troubleshooting

### CUDA Out of Memory
- Reduce `cfg.SOLVER.IMS_PER_BATCH`
- Reduce `cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE`
- Use smaller backbone (R-50 instead of R-101)

### Import Errors
```bash
# Ensure detectron2 installed correctly
pip show detectron2

# Check CUDA compatibility
python -c "import torch; print(torch.cuda.is_available())"
```

### Poor Segmentation Results
- Review HSV color space thresholds
- Check original image quality
- Verify bounding boxes are accurate
- Consider alternative color spaces (LAB, YCrCb)

## Advanced Usage

### Custom Color Segmentation
Modify thresholds in `Predictions_Extract.py`:
```python
# Original
lower_mask = im[:,:,0] > 0.5
upper_mask = im[:,:,0] < 0.9

# For different lighting conditions
lower_mask = im[:,:,0] > 0.4  # More inclusive
upper_mask = im[:,:,0] < 0.95


## Integration with Pipeline

**Input**: Preprocessed images from `../Preprocessing/`
**Output**: Segmented components for `../VisionTransformer/` and `../Histograms/`

Typical workflow:
```
Preprocessing → Labelbox_Detectron2 → VisionTransformer
                                    ↘ Histograms
```



