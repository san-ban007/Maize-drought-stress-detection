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
- YEL (Youngest Expanding Leaf) images
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
```

## Integration with Pipeline

**Input**: Segmented images from `../Labelbox_Detectron2/`
**Output**: Classification predictions and attention visualizations

Pipeline flow:
```
Preprocessing → Detectron2 → VisionTransformer
                           ↘ Histograms
```
