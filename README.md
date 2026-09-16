# CPU-only False Contour Suppression for Quantized Images

A training-free, CPU-only compression-restoration pipeline that suppresses
false contours introduced by direct quantization, targeting low-latency
still-image compression on edge vision systems.

## Overview

Edge devices with visual sensing capabilities must compress images under severe
resource constraints, yet mainstream codecs (JPEG, JPEG2000) and high-performance
deep learning restoration both demand substantial hardware.

We revisit direct quantization---a computationally lightweight scheme whose
artifacts follow a simple mechanism: it merely truncates pixel bit-depth. Around
this scheme, we design a dedicated, training-free false contour suppression
algorithm that forms a CPU-only compression-restoration pipeline:

- Coarse-to-fine detection: localizes artifact-prone smooth regions by
  exploiting the deterministic step structure of quantization bands.
- Linear encoding-decoding: preserves and reconstructs lost gradient
  information to maintain perceptual fidelity.

The pipeline involves no transforms, no entropy coding, and no neural networks.

## Method

The pipeline consists of three stages.

### Stage 1: False Contour Region Detection

- Pre-quantization to target bit-depth: $x \to x - n$ bpp
- Variable thresholding with adaptive window: $s = 2 \lfloor \sqrt{WH}/150 \rfloor + 3$
- Single-iteration SLIC superpixel segmentation
- Adjacency-driven region merging (ARM)
- Region qualification criteria (RQC): neighborhood, equidistant step,
  grayscale richness

### Stage 2: Pre-compression Processing

- Grayscale level grouping into $n$ subgroups
- Grayscale level mapping: $H^j_i = q_i + (j - 1) \cdot 2^n$
- Preserves pixel value lower bound, monotonicity, and gradient direction
- Truncation to compact bitstreams

### Stage 3: Decompression and Reconstruction

- 8-connected component extraction
- Component filtering criteria (CFC)
- Secondary mapping positioning (path-by-path feature determination)
- Secondary mapping:
  - $H2^1_i = 2^n \cdot R2^1_i$
  - $H2^n_i = H2^1_i + 2^n - 1$
  - $H2^j_i = H2^1_i + \lfloor 2^n/(n-1) \rfloor \cdot (j-1)$, for $j = 2,\ldots,n-1$

For color images, the pipeline is applied independently to each RGB channel.

## Experimental Setup

| Item | Details |
|------|---------|
| Proposed method and classical baselines | Raspberry Pi 5 (ARM Cortex-A76, 4 cores, 8 GB RAM), no GPU/NPU |
| Deep learning baselines | NVIDIA Jetson Orin Nano (8 GB RAM), GPU-accelerated |
| Timing | Average of 100 runs after 10 warm-up iterations |

Any speed advantage of the proposed CPU-only method over GPU-accelerated
baselines can thus be attributed to algorithmic efficiency, not hardware.

## Supplementary Material

This supplementary material provides a qualitative visual analysis on the
Kodak24 dataset.

- [`Fig1.pdf`](Fig1.pdf): compression artifact comparison of direct quantization, JPEG, and JPEG2000.
- [`Fig2.pdf`](Fig2.pdf): visual comparison for Setting 1 (direct quantization with traditional decontouring).
- [`Fig3.pdf`](Fig3.pdf): visual comparison for Setting 2 (non-direct quantization with multi-stage traditional restoration).
- [`Fig4.pdf`](Fig4.pdf): visual comparison for Setting 3 (direct quantization with deep learning restoration).
- [`Fig5.pdf`](Fig5.pdf): visual comparison for Setting 4 (non-direct quantization with deep learning restoration).

## Requirements

Python 3.8+, NumPy, Pillow.

`requirements.txt`:

```text
numpy>=1.20
Pillow>=8.0
