# Industrial 3D Defect Detection with PyTorch

An end-to-end PyTorch prototype for defect classification and segmentation in synthetic layer-wise industrial sensor volumes. The project explores leakage-aware validation, 2D defect classification, 2D semantic segmentation, volumetric 3D segmentation, and GPU inference profiling.

> **Scope:** This project uses synthetically generated sensor volumes and simulated defects for methodological experimentation. It does not use proprietary manufacturing data and is not a physically validated production inspection system.

## Why this project

Layer-wise inspection problems create several machine-learning challenges at once: strong correlation between adjacent layers, severe foreground imbalance, small defects, volumetric context, and inference constraints. This repository provides a compact experimental pipeline for working through those issues in PyTorch.

## Pipeline

1. Generate 120 synthetic 3D builds of shape 64 × 128 × 128.
2. Inject ellipsoidal low-intensity defects into defective builds and retain voxel-level masks.
3. Split data at the **build level**, not the slice level, to prevent adjacent layers from leaking across train/validation/test partitions.
4. Train a compact 2D CNN for defect-containing layer classification.
5. Train a 2D U-Net for pixel-level defect localisation.
6. Evaluate false-positive behaviour on normal test layers.
7. Extract 16 × 64 × 64 volumetric patches and train a compact 3D U-Net.
8. Profile 3D inference latency, throughput, parameter count, and parameter memory on GPU.

## Dataset

The synthetic experiment contains:

| Property | Value |
|---|---:|
| Builds | 120 |
| Volume shape | 64 × 128 × 128 |
| Defective builds | 83 (69.17%) |
| Total slices | 7,680 |
| Defective slices | 1,467 (19.10%) |
| Total voxels | 125,829,120 |
| Defect voxels | 116,816 (0.0928%) |

The build-level stratified split is:

| Split | Builds | Normal | Defective |
|---|---:|---:|---:|
| Train | 84 | 26 | 58 |
| Validation | 18 | 6 | 12 |
| Test | 18 | 5 | 13 |

Adjacent layers can be extremely similar; one demonstrated adjacent-layer pair had a pixel correlation of 0.9799. For that reason, all layers from a build remain in the same partition.

## Models and results

### 2D slice classifier

A compact four-block CNN (97,761 parameters) was trained with weighted binary cross-entropy and AdamW.

Locked test-set results:

| Metric | Result |
|---|---:|
| Accuracy | 0.9887 |
| Precision | 0.9880 |
| Recall | 0.9611 |
| F1 | 0.9744 |
| TN / FP / FN / TP | 892 / 3 / 10 / 247 |

### 2D U-Net segmentation

The 2D U-Net contains 1,927,841 parameters and was trained using a combined BCE + soft Dice objective.

Locked positive-slice test results:

| Metric | Result |
|---|---:|
| Loss | 0.0214 |
| Dice | 0.9864 |
| IoU | 0.9852 |

Because evaluating only defect-containing slices can hide operational false positives, the trained model was also evaluated across 895 normal test slices:

| Normal-slice check | Result |
|---|---:|
| Slices with any predicted defect | 11 / 895 |
| Slice false-positive rate | 1.23% |
| Pixel false-positive rate | 0.000075% |

### 3D U-Net

Volumetric patches have shape 1 × 16 × 64 × 64. The compact 3D U-Net contains 339,889 parameters.

Locked 3D test results:

| Metric | Result |
|---|---:|
| Dice | 0.9974 |
| IoU | 0.9948 |

### GPU inference benchmark

Measured on an NVIDIA L40S-48Q using a single 1 × 1 × 16 × 64 × 64 patch after warm-up:

| Metric | Result |
|---|---:|
| Parameters | 339,889 |
| Parameter memory | 1.30 MB |
| Mean latency | 2.22 ms / patch |
| Throughput | 451.16 patches/s |

These timings are hardware- and implementation-specific and should not be interpreted as production edge-device performance.

## Technical choices

**Leakage-aware splitting.** The independent unit is the complete build. Randomly splitting individual layers would allow strongly correlated neighbouring layers from the same synthetic build to appear in different partitions.

**Imbalance-aware evaluation.** Defect voxels account for only 0.0928% of all voxels, so pixel accuracy would be dominated by background. Segmentation performance is therefore reported using Dice and IoU.

**2D before 3D.** The pipeline starts with simpler slice-level classification and localisation before introducing volumetric context. This makes failure modes easier to inspect and provides a baseline against which 3D modelling can be reasoned about.

**Inference profiling.** The final model is profiled for latency, throughput, parameter count, and parameter memory because model quality alone is insufficient when deployment compute is constrained.

## Repository structure

```text
industrial-3d-defect-detection/
├── README.md
├── notebooks/
│   └── Industrial_3D_Defect_Detection.ipynb
├── src/
│   ├── data.py
│   ├── models.py
│   └── metrics.py
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Reproducibility

The experiment was run with Python 3.12.7, PyTorch 2.5.1+cu121, CUDA 12.1, and an NVIDIA L40S-48Q. The synthetic generator uses seed 42.

The notebook is the complete worked experiment. The `src/` modules provide reusable versions of the main data, model, and metric components.

## Limitations

The reported results are intentionally interpreted within the synthetic experimental setting. The generator uses simplified component geometry, intensity distributions, and simulated ellipsoidal defects. The data do not reproduce the full physics, sensor noise, machine variation, material variation, geometry variation, calibration drift, or domain shift encountered in a real industrial inspection system.

High performance therefore demonstrates that the pipeline can learn the constructed synthetic task; it does **not** establish real-world manufacturing performance. A production study would require representative sensor data, physically meaningful ground truth, machine/material/time-aware validation, robustness testing, calibration analysis, and deployment benchmarking on the actual target hardware.

## Author

Hassam Iqbal

- GitHub: https://github.com/hassamiqbal
- Portfolio: https://hassamiqbal.github.io/

## License

MIT License.
