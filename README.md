# Maize Drought Stress Detection

A comprehensive computer vision pipeline for detecting and analyzing drought stress in maize plants using hyperspectral imaging, deep learning, and histogram analysis techniques.

## Overview

This repository contains tools and workflows for automated detection of drought stress in maize plants through image analysis. The system processes side-view camera images of maize plants taken over multiple days, segments individual plant components (Yellow Early Leaf - YEL, and stem), and analyzes pixel intensity distributions to identify stress patterns.

**Dataset**: The complete dataset used in this work is publicly available and can be downloaded from [Zenodo](https://zenodo.org/records/10991581).

## Key Features

- **Automated Image Preprocessing**: Normalization, session division, and patch extraction
- **Object Detection**: Detectron2-based bounding box generation for plant segmentation
- **Vision Transformer Classification**: Fine-tuned ViT models for drought stress classification
- **Histogram Analysis**: Statistical analysis of pixel distributions with Earth Mover's Distance (EMD) calculations
- **Multi-day Tracking**: Temporal analysis of plant stress across experimental trials

## Repository Structure

```
.
├── Preprocessing/           # Image preprocessing and preparation scripts
├── Labelbox_Detectron2/    # Object detection and segmentation using Detectron2
├── VisionTransformer/      # Vision Transformer model training and attention visualization
└── Histograms/             # Histogram generation and statistical analysis
```

## Experimental Setup

The project processes data from greenhouse trials where:
- Each trial contains 4 maize plants
- Plants 2 and 4 are subjected to drought stress
- Plants 1 and 3 serve as well-watered controls
- Images are captured twice daily (morning and evening sessions)
- Multiple trials tracked over 7-17 day periods

## Workflow

### 1. Preprocessing
Raw images from side cameras are processed through multiple stages:
- **Session Division**: Separates daily images into morning and evening scans
- **Image Normalization**: Linear scaling normalization applied to RGB/NIR channels
- **Patch Extraction**: Semi-automatic extraction of individual plant images
- **Background Removal**: Blackout masking to remove adjacent plants

### 2. Object Detection & Segmentation
Using Detectron2 with Labelbox annotations:
- Train Faster R-CNN models to detect YEL (Young Early Leaf) and stem regions
- Generate bounding boxes across all trial images
- Extract segmented plant components for analysis

### 3. Classification & Analysis
- **Vision Transformer**: Fine-tune ViT models for binary classification (drought vs. well-watered)
- **Attention Mapping**: Visualize model attention to understand decision-making
- **Histogram Analysis**: Generate and compare pixel intensity distributions
- **Statistical Metrics**: Calculate mean, std, skewness, and EMD between stress conditions

## Requirements

### Core Dependencies
- Python 3.7+
- PyTorch
- Detectron2
- OpenCV (cv2)
- scikit-learn
- scipy
- numpy
- pandas
- matplotlib

### Additional Tools
- Labelbox (for annotation management)
- transformers (for Vision Transformer)
- PIL/Pillow

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Maize-drought-stress-detection.git
cd Maize-drought-stress-detection

# Install core dependencies
pip install torch torchvision
pip install opencv-python scikit-learn scipy numpy pandas matplotlib
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# Install additional dependencies
pip install labelbox transformers pillow
```

## Usage

### 1. Preprocess Images
```bash
# Divide sessions
python Preprocessing/Session_divide.py

# Normalize images
python Preprocessing/ImageNormalization.py

# Extract patches (use Jupyter notebook)
jupyter notebook Preprocessing/CutImages.ipynb
```

### 2. Train Detection Model
```bash
# Train Detectron2 model (use Jupyter notebook)
jupyter notebook Labelbox_Detectron2/Detectron2_train.ipynb

# Generate bounding boxes
python Labelbox_Detectron2/Bbox_Generate.py

# Extract predictions
python Labelbox_Detectron2/Predictions_Extract.py
```

### 3. Run Classification
```bash
# Fine-tune Vision Transformer
jupyter notebook VisionTransformer/Vision_Transformer_FineTuning.ipynb

# Generate attention maps
jupyter notebook VisionTransformer/Vision_Transformer_attn_map.ipynb
```

### 4. Analyze Histograms
```bash
# Generate daily histograms
jupyter notebook Histograms/Histograms_per_day.ipynb

# Plot comparative histograms
jupyter notebook Histograms/Plot_histograms_of_well_watered_and_drought_stressed_in_one_image.ipynb

# Calculate EMD
jupyter notebook Histograms/EMD.ipynb
```

## Configuration

Each script contains hardcoded paths that need to be updated for your environment:
- Trial numbers (e.g., '005', '008')
- Root directories for data storage
- API keys for Labelbox integration
- Model checkpoint paths

## Data Format

Expected directory structure for input data:
```
Trial_XXX/
├── TrialXXX/
│   ├── y20mXXdXX/
│   │   └── [raw images]
├── Sessions/
│   ├── y20mXXdXX/
│   │   ├── Session1/
│   │   └── Session2/
└── Patches/
    ├── Plant1/
    ├── Plant2/
    ├── Plant3/
    └── Plant4/
```

## Output

The pipeline generates:
- Normalized and preprocessed images
- Bounding box visualizations
- Segmented plant components (YEL and stem)
- Classification predictions
- Attention heatmaps
- Histogram plots and statistical comparisons
- EMD distance metrics

## Model Performance

The system can detect drought stress by analyzing:
- Pixel intensity distribution changes
- Color space transformations (HSV)
- Temporal progression of stress symptoms
- Spatial attention patterns from ViT

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Citation

If you use this code or dataset in your research, please cite the associated paper:

**Journal Paper**: [Add journal citation and DOI here]

**Dataset**: 
```bibtex
@dataset{maize_drought_dataset,
  author       = {[Authors]},
  title        = {Maize Drought Stress Detection Dataset},
  year         = {2024},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10991581},
  url          = {https://zenodo.org/records/10991581}
}
```

## License

[Add your license information here]

## Authors

- Sanjana Srabanti

## Acknowledgments

This work was conducted as part of plant phenotyping research using greenhouse imaging systems.

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainers.
