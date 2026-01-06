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

**Output**:
```
Comparison_Plots/
├── Trial005_full_comparison.png
├── Trial005_mean_trends.png
├── Trial005_std_trends.png
└── Trial005_statistics_table.pdf
```


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
 

