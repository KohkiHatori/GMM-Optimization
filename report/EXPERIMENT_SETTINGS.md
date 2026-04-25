# Experiment Settings

This document outlines the hardware specifications and compilation settings used for the GMM Optimization experiments.

## Hardware Specifications

### CPU Environment (Serial and OpenMP)
The Serial and OpenMP versions were tested on a high-performance workstation with the following specifications:

| Parameter | Value |
| :--- | :--- |
| **Model Name** | Intel(R) Core(TM) i9-14900 |
| **Architecture** | x86_64 |
| **Total CPUs (Threads)** | 32 |
| **Cores per Socket** | 24 |
| **Threads per Core** | 1 |
| **Base Clock Speed** | 2.65 GHz |
| **Max Turbo Boost** | 5.80 GHz |
| **L1d/L1i Cache** | 48K / 32K |
| **L2 Cache** | 2048K |
| **L3 Cache** | 36864K |
| **Memory Order** | Little Endian |

### GPU Environment (CUDA)
The CUDA version was executed on an NVIDIA Tesla P100.

| Parameter | Value |
| :--- | :--- |
| **GPU Model** | NVIDIA Tesla P100 (Pascal Architecture) |
| **Compute Capability** | 6.0 |

---

## Compilation Flags

The projects were compiled using the following flags as defined in the `Makefile`:

### Serial Version
- **Compiler**: `gcc`
- **Flags**: `-O3 -ffast-math -Wall`
- **Libraries**: `-lm` (Math library)

### OpenMP Version
- **Compiler**: `gcc`
- **Flags**: `-O3 -ffast-math -Wall -fopenmp`
- **Libraries**: `-lm` (Math library)

### CUDA Version
- **Compiler**: `nvcc`
- **Flags**: `-O3 -arch=sm_60`
- **Libraries**: `-lm` (Math library)

---

## Execution Environment
- **Operating System**: Linux (CentOS/RHEL based environment typically found on signal clusters)
- **OpenMP Threads**: Tested across various thread counts (e.g., 2, 4, 8, 16, 24, 32) as specified in the results directory.
