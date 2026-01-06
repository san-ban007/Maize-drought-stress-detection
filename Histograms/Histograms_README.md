# Histograms

Statistical analysis module for comparing pixel intensity distributions between drought-stressed and well-watered maize plants using histogram analysis and Earth Mover's Distance (EMD) metrics.

## Overview

This module implements histogram-based analysis to:
1. **Generate pixel intensity histograms** for each plant on each day
2. **Compare distributions** between drought-stressed and control plants
3. **Calculate statistical metrics** (mean, std, skewness, kurtosis)
4. **Compute Earth Mover's Distance** to quantify distribution differences
5. **Visualize temporal patterns** of stress progression

The approach is based on the hypothesis that drought stress causes measurable shifts in pixel intensity distributions, particularly in segmented plant tissues (YEL and stem).

## Theoretical Background

### Histogram Analysis
Plant stress manifests as color changes:
- **Chlorosis**: Yellowing due to chlorophyll degradation
- **Necrosis**: Browning/darkening of dead tissue
- **Wilting**: Changes in tissue reflectance

These physiological changes alter pixel intensity distributions:
```
Well-watered:   [Strong peak in green region]
Drought-stressed: [Shifted toward yellow/brown, broader distribution]
```

### Earth Mover's Distance (EMD)
Also known as Wasserstein distance, EMD measures the minimum "work" required to transform one distribution into another:

```
EMD(P, Q) = minimum cost to move mass from distribution P to Q
```

**Advantages**:
- Captures shape and position differences
- Robust to small variations
- Interpretable as "distance" between distributions
- Works well for comparing histograms

## Files

### `Histograms_per_day.ipynb`
**Purpose**: Generate and save histograms plus statistical metrics for each plant on each day

**Workflow**:
```
For each Trial:
    For each Date:
        For each Plant:
            For each Scan (morning/evening):
                1. Load segmented images (YEL or Stem)
                2. Extract pixel intensities
                3. Generate histogram
                4. Calculate statistics
                5. Save histogram plot
                6. Save statistics to CSV
```

**Statistical Metrics Computed**:
- **Mean**: Average pixel intensity
- **Standard Deviation**: Spread of intensity values
- **Skewness**: Asymmetry of distribution
- **Kurtosis**: "Tailedness" of distribution
- **Median**: Middle value
- **Mode**: Most frequent value
- **Percentiles**: 25th, 75th for quartile analysis

**Histogram Configuration**:
```python
# Typical parameters
bins = 256  # One bin per intensity level (0-255)
range = (0, 255)  # Full intensity range
density = True  # Normalize to probability distribution
```

**Channel Analysis**:
Can analyze individual or combined channels:
- **Blue channel**: Least affected by chlorophyll
- **Green channel**: Most sensitive to chlorosis
- **Red/NIR channel**: Useful for stress detection
- **Combined RGB**: Overall color information

**Output Structure**:
```
Histograms/
├── Trial005/
│   ├── Plant1/
│   │   ├── y20m10d01_scan1_histogram.png
│   │   ├── y20m10d01_scan1_stats.csv
│   │   ├── y20m10d02_scan1_histogram.png
│   │   └── ...
│   ├── Plant2/  # Drought stressed
│   ├── Plant3/
│   └── Plant4/  # Drought stressed
└── statistics_summary.csv  # All metrics across trial
```

**Statistics CSV Format**:
```csv
Date,Plant,Scan,Mean,Std,Skewness,Kurtosis,Median,Q25,Q75
y20m10d01,Plant1,1,145.3,25.6,-0.4,2.1,148,130,165
y20m10d01,Plant2,1,132.1,28.9,-0.2,2.4,135,115,152
...
```

---

### `Plot_histograms_of_well_watered_and_drought_stressed_in_one_image.ipynb`
**Purpose**: Create comprehensive visualizations comparing drought vs. control throughout trial

**Visualization Types**:

#### 1. Combined Histogram Plot
```
[Day 1]  [Day 2]  [Day 3]  ...  [Day N]
 Ctrl     Ctrl     Ctrl          Ctrl    (Plant 1/3 overlaid)
 Drght    Drght    Drght         Drght   (Plant 2/4 overlaid)
```

Features:
- Side-by-side comparison per day
- Overlaid well-watered (blue) and drought (red)
- Common y-axis for fair comparison
- Vertical lines for mean values

#### 2. Statistical Trends Over Time
Line plots showing temporal changes:
```python
# Mean intensity over days
plt.plot(days, control_means, 'b-', label='Control')
plt.plot(days, drought_means, 'r-', label='Drought')

# Standard deviation trends
# Skewness evolution
# Etc.
```

#### 3. Distribution Evolution Animation
Optional: Create GIF showing day-by-day histogram changes

**Key Insights from Visualizations**:
- **Separation timing**: When do distributions diverge?
- **Magnitude of shift**: How much do histograms differ?
- **Recovery patterns**: If rewatered, how quickly do distributions normalize?
- **Consistency**: Do both drought plants show similar patterns?

**Plot Configuration**:
```python
# Styling
figsize = (20, 12)  # Large enough for all days
dpi = 300  # High resolution
colors = {
    'control': '#2E86AB',      # Blue
    'drought': '#A23B72',      # Red
    'control_std': '#A7C4D6',  # Light blue
    'drought_std': '#E6B8D1'   # Light red
}

# Layout
subplot_rows = 2
subplot_cols = len(dates) // 2
sharex = True
sharey = True
```

**Output**:
```
Comparison_Plots/
├── Trial005_full_comparison.png
├── Trial005_mean_trends.png
├── Trial005_std_trends.png
└── Trial005_statistics_table.pdf
```

---

### `EMD.ipynb`
**Purpose**: Calculate Earth Mover's Distance between drought and control histograms

**Mathematical Foundation**:
```
EMD(P, Q) = min Σᵢ Σⱼ fᵢⱼ · dᵢⱼ
            subject to:
            Σⱼ fᵢⱼ = pᵢ  (supply constraint)
            Σᵢ fᵢⱼ = qⱼ  (demand constraint)
            fᵢⱼ ≥ 0      (non-negativity)
```

Where:
- P, Q: Histograms to compare
- fᵢⱼ: Flow from bin i to bin j
- dᵢⱼ: Distance between bins i and j

**Implementation**:
```python
from scipy.stats import wasserstein_distance

# For each day
control_hist = get_histogram(plant1_images + plant3_images)
drought_hist = get_histogram(plant2_images + plant4_images)

emd_value = wasserstein_distance(
    u_values=bin_centers,
    v_values=bin_centers,
    u_weights=control_hist,
    v_weights=drought_hist
)
```

**EMD Analysis**:

1. **Per-Day EMD**:
```python
emds = []
for day in trial_days:
    control_dist = combine_histograms(plant1_day, plant3_day)
    drought_dist = combine_histograms(plant2_day, plant4_day)
    emd = wasserstein_distance(control_dist, drought_dist)
    emds.append(emd)
```

2. **EMD Trends**:
```python
plt.plot(days, emds, 'o-')
plt.xlabel('Days since drought initiation')
plt.ylabel('EMD (intensity units)')
plt.title('Earth Mover\'s Distance: Drought vs. Control')
```

**Interpretation**:
- **Low EMD (< 5)**: Distributions very similar (early stage or recovery)
- **Medium EMD (5-15)**: Moderate stress visible
- **High EMD (> 15)**: Severe stress, clearly separated distributions
- **Increasing trend**: Progressive stress development
- **Decreasing trend**: Recovery or acclimation

**Channel-Specific EMD**:
```python
emd_blue = wasserstein_distance(control_blue, drought_blue)
emd_green = wasserstein_distance(control_green, drought_green)
emd_red = wasserstein_distance(control_red, drought_red)

# Most sensitive channel?
most_sensitive = max(emd_blue, emd_green, emd_red)
```

**Statistical Testing**:
```python
# Bootstrap confidence intervals
bootstrap_emds = []
for _ in range(1000):
    # Resample pixels
    control_sample = resample(control_pixels)
    drought_sample = resample(drought_pixels)
    emd = wasserstein_distance(control_sample, drought_sample)
    bootstrap_emds.append(emd)

ci_lower = np.percentile(bootstrap_emds, 2.5)
ci_upper = np.percentile(bootstrap_emds, 97.5)
```

**Output**:
```
EMD_Results/
├── Trial005_EMD_per_day.csv
├── Trial005_EMD_trends.png
├── Trial005_channel_comparison.png
└── Trial005_statistical_tests.txt
```

**CSV Format**:
```csv
Date,Day,EMD_Blue,EMD_Green,EMD_Red,EMD_Combined,CI_Lower,CI_Upper
y20m10d01,1,3.2,4.5,2.8,3.5,2.9,4.1
y20m10d02,2,5.1,7.2,4.3,5.5,4.8,6.2
y20m10d03,3,8.3,11.5,7.9,9.2,8.5,9.9
...
```

---

## Dependencies

```bash
# Core libraries
pip install numpy pandas
pip install matplotlib seaborn
pip install scipy

# Image processing
pip install opencv-python
pip install scikit-image
pip install pillow

# Statistics
pip install statsmodels
```

## Configuration

### Input Paths
```python
# Histograms_per_day.ipynb
trial_num = '005'
root_path = '/path/to/Trial005/Patches'
scan = '1'  # 1=morning, 2=evening
tissue_type = 'YEL'  # or 'STEM'
```

### Histogram Parameters
```python
bins = 256
range = (0, 255)
density = True  # Normalize histograms
alpha = 0.6  # Transparency for overlays
```

### Color Channels
```python
# Analyze specific channels
channels = {
    'blue': 0,
    'green': 1,
    'red': 2
}

# Or combined
use_grayscale = False
```

## Usage

### Generate Daily Histograms
```bash
jupyter notebook Histograms_per_day.ipynb

# Follow cells:
1. Set trial and paths
2. Load segmented images
3. Extract pixel values
4. Generate histograms
5. Calculate statistics
6. Save outputs
```

### Create Comparison Plots
```bash
jupyter notebook Plot_histograms_of_well_watered_and_drought_stressed_in_one_image.ipynb

# Execute:
1. Load all daily histograms
2. Organize by treatment
3. Create grid layout
4. Plot comparisons
5. Add statistical annotations
6. Export high-res figure
```

### Calculate EMD
```bash
jupyter notebook EMD.ipynb

# Run:
1. Load histograms for control and drought
2. Compute EMD per day
3. Analyze trends
4. Perform statistical tests
5. Generate visualizations
```

## Output Interpretation

### Histogram Shapes

**Normal (Well-watered)**:
```
   ^
   |     ***
   |   ******
   |  ********
   | **********
   |************
   +-------------> Intensity
   Low    High
```
- Single peak (unimodal)
- Peak in green region (120-160)
- Narrow spread
- Symmetric or slight negative skew

**Drought-Stressed**:
```
   ^
   |   ***
   |  *****    **
   | ******* ****
   |*************
   +-------------> Intensity
   Low    High
```
- Broader distribution
- Peak shifted left (lower intensity)
- Possible bimodal (healthy + stressed tissue)
- Positive skew (tail toward higher values)

### Statistical Significance

**Mean Decrease**:
- Control: μ ≈ 145
- Drought: μ ≈ 125
- Indicates darker pixels (less chlorophyll)

**Increased Variability**:
- Control: σ ≈ 20
- Drought: σ ≈ 35
- More heterogeneous tissue condition

**Skewness Changes**:
- Control: skew ≈ -0.3 (slight left tail)
- Drought: skew ≈ +0.5 (right tail from necrosis)

### EMD Thresholds

Based on empirical observations:
- **0-5**: No significant stress
- **5-10**: Early/mild stress
- **10-20**: Moderate stress
- **20+**: Severe stress

## Advanced Analysis

### Multi-Channel Integration
```python
# Weighted EMD across channels
weights = {
    'blue': 0.2,
    'green': 0.5,  # Most informative
    'red': 0.3
}

combined_emd = (
    weights['blue'] * emd_blue +
    weights['green'] * emd_green +
    weights['red'] * emd_red
)
```

### Temporal Smoothing
```python
from scipy.signal import savgol_filter

# Smooth EMD trends
window_length = 5
polyorder = 2
smoothed_emd = savgol_filter(emd_values, window_length, polyorder)
```

### Hypothesis Testing
```python
from scipy.stats import mannwhitneyu

# Test if control and drought distributions differ
stat, pvalue = mannwhitneyu(
    control_pixels,
    drought_pixels,
    alternative='two-sided'
)

print(f"U-statistic: {stat}")
print(f"p-value: {pvalue}")
# Significant if p < 0.05
```

### Correlation Analysis
```python
import pandas as pd

# Correlate EMD with plant measurements
df = pd.DataFrame({
    'EMD': emd_values,
    'Plant_Height': height_measurements,
    'Soil_Moisture': moisture_readings,
    'Temperature': temp_data
})

correlations = df.corr()
```

### Machine Learning Classification
```python
from sklearn.ensemble import RandomForestClassifier

# Features from histograms
features = np.column_stack([
    means, stds, skewnesses, kurtoses, emds
])
labels = [0]*len(control) + [1]*len(drought)

clf = RandomForestClassifier()
clf.fit(features, labels)
accuracy = clf.score(test_features, test_labels)
```

## Validation

### Cross-Trial Validation
```python
# Train on Trial 005, test on Trial 008
train_features = extract_features(trial_005)
test_features = extract_features(trial_008)

model.fit(train_features, train_labels)
test_accuracy = model.score(test_features, test_labels)
```

### Temporal Consistency
```python
# Check if patterns repeat across trials
for trial in [005, 006, 007, 008]:
    emd_trend = calculate_emd_trend(trial)
    peak_stress_day = np.argmax(emd_trend)
    print(f"Trial {trial}: Peak stress on Day {peak_stress_day}")
```

## Troubleshooting

### Issue: Histograms Look Similar
**Possible Causes**:
- Too early in stress progression
- Insufficient drought severity
- Segmentation includes background
- Wrong color channel analyzed

**Solutions**:
- Analyze later time points
- Check experimental protocol
- Improve segmentation
- Try green channel specifically

### Issue: EMD Values Unreasonably High/Low
**Possible Causes**:
- Normalization issues
- Incorrect bin edges
- Mixed tissue types

**Solutions**:
- Verify histogram normalization
- Use consistent bin ranges
- Separate YEL and stem analysis

### Issue: Large Variance Across Replicates
**Possible Causes**:
- Plant-to-plant variation
- Imaging inconsistencies
- Segmentation errors

**Solutions**:
- Use median instead of mean
- Apply stricter quality control
- Increase sample size

## Performance Optimization

### Memory Efficiency
```python
# Process images in batches
def process_in_batches(image_paths, batch_size=100):
    histograms = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        hist = compute_batch_histogram(batch)
        histograms.append(hist)
    return np.sum(histograms, axis=0)
```

### Parallel Processing
```python
from multiprocessing import Pool

def compute_histogram_parallel(image_paths, n_cores=4):
    with Pool(n_cores) as p:
        histograms = p.map(compute_single_histogram, image_paths)
    return np.sum(histograms, axis=0)
```

## Integration with Pipeline

**Input Sources**:
- Segmented images from `Labelbox_Detectron2/`
- Preprocessed images from `Preprocessing/`

**Complementary Analysis**:
- Histogram analysis provides quantitative stress metrics
- Vision Transformers provide classification accuracy
- Combined approach increases confidence

**Workflow Integration**:
```
Detectron2 Segmentation
         ↓
    Histogram Analysis ← → Vision Transformer
         ↓                        ↓
    EMD Metrics            Attention Maps
         ↓                        ↓
         └─── Combined Report ────┘
```

## Biological Validation

### Expected Patterns

**Drought Initiation** (Days 1-3):
- Minimal histogram changes
- EMD < 5

**Visible Stress** (Days 4-7):
- Histogram separation begins
- EMD 5-15
- Mean intensity decreases

**Severe Stress** (Days 8+):
- Clear histogram separation
- EMD > 15
- Increased variability

### Physiological Correlation
- **Chlorophyll loss**: Green channel shift to lower values
- **Anthocyanin accumulation**: Red channel increase
- **Tissue death**: Bimodal distributions

## Citation

If using EMD for plant stress analysis:
```bibtex
@article{rubner2000emd,
  title={The earth mover's distance as a metric for image retrieval},
  author={Rubner, Yossi and Tomasi, Carlo and Guibas, Leonidas J},
  journal={International journal of computer vision},
  volume={40},
  number={2},
  pages={99--121},
  year={2000}
}
```

## References

- [SciPy wasserstein_distance Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)
- [Earth Mover's Distance Tutorial](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)

## Related Documentation

- Main README: `../README.md`
- Vision Transformer: `../VisionTransformer/README.md`
- Object Detection: `../Labelbox_Detectron2/README.md`
