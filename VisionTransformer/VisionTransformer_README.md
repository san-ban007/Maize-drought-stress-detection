# VisionTransformer

Deep learning classification pipeline using Vision Transformer (ViT) models to classify maize plants as drought-stressed or well-watered, with integrated attention visualization for model interpretability.

## Overview

This module implements:
1. **Fine-tuning** pre-trained Vision Transformer models on maize drought stress classification
2. **Attention Map Visualization** to understand which image regions influence model decisions
3. **Binary Classification** distinguishing between drought-stressed and well-watered plants

The approach leverages transfer learning from ImageNet-pretrained ViT models, adapting them to the specific task of plant stress detection through fine-tuning on segmented maize plant images.

## Experimental Context

**Classification Task**: Binary classification
- **Class 0**: Well-watered (Control) - Plants 1 and 3
- **Class 1**: Drought-stressed - Plants 2 and 4

**Input Data**: Segmented plant images from `Labelbox_Detectron2/`
- YEL (Young Early Leaf) images
- Stem images
- Full plant patches

## Files

### `Vision_Transformer_FineTuning.ipynb`
**Purpose**: Train and fine-tune Vision Transformer models for drought stress classification

**Model Architecture**:
```python
# Typical ViT configuration
Model: Vision Transformer (ViT-B/16 or ViT-L/16)
Input Size: 224x224 pixels (resized from originals)
Patch Size: 16x16 pixels
Number of Classes: 2 (drought vs. well-watered)
Pre-training: ImageNet-21k or ImageNet-1k
```

**Key Components**:

1. **Data Loading & Augmentation**
   ```python
   transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(10),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
   ])
   ```

2. **Model Configuration**
   - Load pre-trained ViT from HuggingFace transformers
   - Replace classification head for binary output
   - Optionally freeze backbone layers for faster training

3. **Training Loop**
   - Cross-entropy loss for classification
   - AdamW optimizer with weight decay
   - Learning rate scheduling (cosine or step decay)
   - Early stopping based on validation accuracy

4. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion matrix
   - ROC curve and AUC

**Workflow**:
```
1. Load Dataset
   ├─ Organize by plant and date
   ├─ Split into train/validation/test
   └─ Apply augmentations

2. Initialize Model
   ├─ Load pre-trained ViT
   ├─ Modify classification head
   └─ Set trainable parameters

3. Training
   ├─ Forward pass
   ├─ Compute loss
   ├─ Backward pass
   ├─ Update weights
   └─ Validate after each epoch

4. Evaluation
   ├─ Test set predictions
   ├─ Calculate metrics
   └─ Visualize results

5. Save Model
   └─ Export trained weights
```

**Hyperparameters**:
```python
batch_size = 16
learning_rate = 1e-4
epochs = 20
weight_decay = 0.01
patience = 5  # Early stopping
```

**Expected Performance**:
- Training Accuracy: 85-95%
- Validation Accuracy: 80-90%
- Test Accuracy: 75-85% (depending on trial variability)

---

### `Vision_Transformer_attn_map.ipynb`
**Purpose**: Visualize attention maps to interpret model decision-making

**Attention Mechanism**:
Vision Transformers use multi-head self-attention to:
- Identify important image patches
- Capture long-range dependencies
- Focus on discriminative plant features

**Visualization Techniques**:

1. **Attention Rollout**
   - Aggregates attention across all layers
   - Shows cumulative attention flow
   - Highlights most influential patches

2. **Attention Map Overlay**
   - Superimposes attention weights on original image
   - Uses heatmap coloring (cool to warm)
   - Reveals spatial focus areas

3. **Per-Head Attention**
   - Visualizes individual attention heads
   - Shows diverse feature detection
   - Helps understand multi-head specialization

**Workflow**:
```python
1. Load Trained Model
   └─ Import fine-tuned ViT weights

2. Select Test Images
   ├─ Drought-stressed examples
   └─ Well-watered examples

3. Extract Attention Weights
   ├─ Hook into transformer layers
   ├─ Capture attention matrices
   └─ Aggregate across heads/layers

4. Generate Visualizations
   ├─ Attention rollout maps
   ├─ Overlay on original image
   └─ Side-by-side comparisons

5. Interpret Results
   ├─ Identify discriminative regions
   ├─ Compare drought vs. control
   └─ Validate biological relevance
```

**Key Functions**:

`get_attention_maps(model, image)`:
- Extracts raw attention weights from model
- Returns: [num_layers, num_heads, num_patches, num_patches]

`attention_rollout(attention_weights)`:
- Computes cumulative attention across layers
- Applies identity matrix addition for gradient flow
- Returns: [num_patches, num_patches]

`visualize_attention(image, attention_map)`:
- Resizes attention to match image dimensions
- Applies colormap (e.g., 'jet', 'viridis')
- Overlays on original image with transparency

**Biological Insights**:
- Model focuses on **leaf tips** for stress indicators
- Attention to **leaf margins** (wilting/curling)
- Emphasis on **color variations** (chlorosis)
- Lower attention to **stems** and **background**

**Example Output**:
```
Original Image | Attention Map | Overlay
[Plant Image]  | [Heatmap]     | [Combined]

Key Findings:
- High attention on YEL (drought-sensitive)
- Focus on discolored leaf regions
- Minimal attention to soil/pot
```

---

## Dataset Organization

### Directory Structure
```
Dataset/
├── train/
│   ├── drought/
│   │   ├── Plant2_y20m10d01_0001.jpg
│   │   ├── Plant2_y20m10d02_0001.jpg
│   │   └── Plant4_...
│   └── well_watered/
│       ├── Plant1_y20m10d01_0001.jpg
│       ├── Plant3_y20m10d02_0001.jpg
│       └── ...
├── val/
│   ├── drought/
│   └── well_watered/
└── test/
    ├── drought/
    └── well_watered/
```

### Data Split Strategy
```python
# Temporal split (recommended)
Train: Days 1-10
Val:   Days 11-13
Test:  Days 14-17

# Plant-wise split (alternative)
Train: Plants 1,2
Val:   Plant 3
Test:  Plant 4
```

### Class Balance
- Ensure equal drought/well-watered samples
- Use weighted loss if imbalanced
- Apply oversampling/undersampling if needed

## Model Variants

### ViT-Base (Recommended)
```python
model_name = 'google/vit-base-patch16-224'
parameters = 86M
training_time = ~2 hours (GPU)
accuracy = 80-85%
```

### ViT-Large
```python
model_name = 'google/vit-large-patch16-224'
parameters = 304M
training_time = ~6 hours (GPU)
accuracy = 82-88%
note = "Requires more data and GPU memory"
```

### ViT-Tiny (Fast baseline)
```python
model_name = 'WinKawaks/vit-tiny-patch16-224'
parameters = 5M
training_time = ~30 minutes (GPU)
accuracy = 70-75%
note = "Good for prototyping"
```

## Dependencies

```bash
# Core libraries
pip install torch torchvision
pip install transformers  # HuggingFace
pip install timm  # PyTorch Image Models

# Visualization
pip install matplotlib seaborn
pip install opencv-python
pip install scikit-image

# Utilities
pip install numpy pandas
pip install scikit-learn
pip install tqdm
```

## Configuration

### Import Pre-trained Model
```python
from transformers import ViTForImageClassification, ViTFeatureExtractor

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=2,
    ignore_mismatched_sizes=True
)
```

### Data Loaders
```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)
```

### Training Configuration
```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)
```

## Usage

### Training
```bash
# Open notebook
jupyter notebook Vision_Transformer_FineTuning.ipynb

# Execute cells in order:
1. Import libraries
2. Load and prepare dataset
3. Initialize model
4. Define training loop
5. Train model
6. Evaluate performance
7. Save model weights
```

**Training Tips**:
- Start with frozen backbone, unfreeze gradually
- Monitor validation loss for overfitting
- Use learning rate warmup (first few epochs)
- Save checkpoints every epoch
- Log to TensorBoard for visualization

### Attention Visualization
```bash
# Open notebook
jupyter notebook Vision_Transformer_attn_map.ipynb

# Execute cells:
1. Load trained model
2. Select test images
3. Extract attention weights
4. Generate attention maps
5. Create visualizations
6. Interpret results
```

## Output

### Model Checkpoints
```
models/
├── vit_drought_epoch_10.pth
├── vit_drought_epoch_15.pth
├── vit_drought_best.pth          # Best validation accuracy
└── vit_drought_final.pth
```

### Training Logs
```
logs/
├── training_loss.csv
├── validation_metrics.csv
├── confusion_matrix.png
└── roc_curve.png
```

### Attention Maps
```
attention_maps/
├── drought_plant2_day10.png
├── drought_plant4_day12.png
├── control_plant1_day10.png
└── control_plant3_day12.png
```

## Performance Analysis

### Evaluation Metrics

**Confusion Matrix**:
```
                Predicted
                Drought | Well-watered
Actual Drought     TP   |     FN
     Well-watered  FP   |     TN
```

**Metrics**:
- Accuracy = (TP + TN) / Total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

### Cross-Validation
```python
# K-Fold on different trials
for trial in [005, 006, 007, 008]:
    train_on_other_trials()
    test_on_current_trial()
    record_metrics()
```

### Error Analysis
Common misclassifications:
1. Early drought stages (subtle symptoms)
2. Recovery phase (after rewatering)
3. Natural leaf variations
4. Lighting/shadow effects

## Optimization Strategies

### Data Augmentation
```python
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    # ... normalization
])
```

### Regularization Techniques
- Dropout in classification head
- Weight decay (L2 regularization)
- Stochastic depth (layer dropping)
- Mixup or CutMix augmentation

### Transfer Learning Strategies

**Strategy 1: Feature Extraction**
```python
# Freeze all layers except classifier
for param in model.vit.parameters():
    param.requires_grad = False
# Only train classifier
```

**Strategy 2: Fine-tuning**
```python
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True
# Use lower learning rate
```

**Strategy 3: Gradual Unfreezing**
```python
# Epochs 1-5: Only classifier
# Epochs 6-10: Last 4 transformer blocks
# Epochs 11+: All layers
```

## Interpretability

### Attention Pattern Analysis

**Drought-Stressed Plants**:
- High attention on leaf tips (wilting)
- Focus on discolored regions (chlorosis)
- Emphasis on leaf margins (curling)

**Well-Watered Plants**:
- Broader attention distribution
- Focus on overall plant structure
- Less emphasis on specific stress indicators

### Gradient-Based Methods
Supplement attention maps with:
- Grad-CAM
- Integrated Gradients
- Saliency Maps

## Troubleshooting

### Low Accuracy
- **Insufficient Data**: Collect more labeled examples
- **Class Imbalance**: Apply weighted loss or resampling
- **Poor Augmentation**: Review augmentation strategies
- **Learning Rate**: Try lower/higher values

### Overfitting
- Increase dropout rate
- Add more augmentation
- Reduce model capacity (use ViT-Base instead of ViT-Large)
- Implement early stopping

### Attention Maps Not Informative
- Check if model is actually trained (not random)
- Try different attention aggregation methods
- Visualize multiple layers separately
- Compare with baseline (random model)

### GPU Memory Issues
```python
# Reduce batch size
batch_size = 8  # instead of 16

# Use gradient accumulation
accumulation_steps = 2
# Effective batch size = 8 * 2 = 16

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

## Advanced Techniques

### Ensemble Methods
```python
# Train multiple models
models = [vit_base, vit_large, resnet50]

# Average predictions
predictions = [model(image) for model in models]
final_pred = torch.mean(torch.stack(predictions), dim=0)
```

### Multi-Task Learning
```python
# Simultaneous classification tasks
tasks = {
    'drought_stress': 2 classes,
    'growth_stage': 4 classes,
    'plant_health': 3 classes
}
# Share backbone, separate heads
```

### Temporal Modeling
```python
# Sequence of images over time
# Use LSTM or Transformer on ViT features
sequence = [day1_features, day2_features, ..., dayN_features]
temporal_model = LSTM(input_features, hidden_dim)
```

## Integration with Pipeline

**Input**: Segmented images from `../Labelbox_Detectron2/`
**Output**: Classification predictions and attention visualizations

Pipeline flow:
```
Preprocessing → Detectron2 → VisionTransformer
                           ↘ Histograms
```

**Complementary Analysis**:
- ViT classifications validate histogram-based stress detection
- Attention maps identify which features histograms should focus on
- Combined approach increases robustness

## Citation

```bibtex
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  journal={ICLR},
  year={2021}
}
```

## References

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Attention Visualization Tutorial](https://jacobgil.github.io/deeplearning/vision-transformer-explainability)

## Related Documentation

- Main README: `../README.md`
- Object Detection: `../Labelbox_Detectron2/README.md`
- Histogram Analysis: `../Histograms/README.md`
