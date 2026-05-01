# Accelerating Regularized Attention Kernel Regression for Spectrum Cartography

This repository provides code for regularized attention kernel regression and an efficient learning-based solver for spectrum cartography. It addresses the ill-conditioning of attention-based exponential kernels. Numerical experiments demonstrate accelerated convergence and accurate radio map reconstruction. For more details, please refer to our paper "Accelerating Regularized Attention Kernel Regression for Spectrum Cartography" (https://doi.org/10.48550/arXiv.2604.25138)

---

## Installation
The following software and libraries are required:
- Python 3.11
- PyTorch 2.5.1
- CVXPY
- NVIDIA Sionna RT

## Repository Structure
This repository contains the following example scripts:

- `example_spectrum_of_attention_kernel_in_model.py`  
  Demonstrates the spectral properties of the attention kernel used in the model.

- `numerical_example_in_algorithm_with_sionna.py`  
  Provides a numerical example illustrating the radio map reconstruction process in the algorithm using NVIDIA Sionna.

- `sc_radiomap_kernel_in_introduction_with_sionna_munich.py`  
  Implements kernel-based radio map reconstruction in a realistic Munich urban environment simulated with NVIDIA Sionna.

## Usage

### Running the Python script

Python scripts, such as `numerical_example_in_algorithm_with_sionna.py`, can be executed in two ways:

#### 1. Direct Execution
Open the file in a Python IDE (e.g., PyCharm) and run it directly.

#### 2. Command-Line Execution
From a terminal, navigate to the script directory and run:

```bash
python numerical_example_in_algorithm_with_sionna.py
```

## Citation
```bibtex
@article{tao2026accelerating,
  title={Accelerating Regularized Attention Kernel Regression for Spectrum Cartography},
  author={Tao, Liping and Tan, Chee Wei},
  journal={arXiv preprint arXiv:2604.25138},
  year={2026}
}
```
