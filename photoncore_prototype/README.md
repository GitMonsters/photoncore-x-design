# PhotonCore-X Prototype

Simulation and prototyping framework for PhotonCore-X photonic AI accelerator.

## Features

- **MZI Mesh Simulation**: Full simulation of programmable Mach-Zehnder interferometer meshes
- **Clements Decomposition**: Decompose arbitrary unitary matrices into MZI phases
- **Optical Matrix-Vector Multiply**: Simulate optical linear algebra operations
- **All-Optical Nonlinearities**: ReLU, sigmoid, softmax without O-E-O conversion
- **WDM Parallel Processing**: 128-channel wavelength-multiplexed computation
- **Calibration System**: Auto-calibration with ML acceleration
- **Neural Network SDK**: PyTorch-like API for optical neural networks

## Installation

```bash
cd photoncore_prototype
pip install -r requirements.txt
```

## Quick Start

```python
from src.photoncore import PhotonCoreSimulator, create_optical_mlp
from src.clements import random_unitary
import numpy as np

# Create simulator
sim = PhotonCoreSimulator(n_ports=64)

# Load a matrix
U = random_unitary(64)
sim.load_matrix(U)

# Matrix-vector multiply
x = np.random.randn(64)
y = sim.matmul(x)

# Create neural network
network = create_optical_mlp([784, 256, 128, 10])
output = network(np.random.randn(784))
```

## Project Structure

```
photoncore_prototype/
├── src/
│   ├── mzi.py           # MZI mesh simulation
│   ├── clements.py      # Matrix decomposition
│   ├── optical_mvm.py   # Optical matrix multiply
│   ├── calibration.py   # Calibration system
│   ├── photoncore.py    # Main SDK
│   └── wdm.py           # WDM system
├── benchmarks/
│   └── benchmark_suite.py
├── examples/
│   └── mnist_inference.py
└── tests/
```

## Running Benchmarks

```bash
python benchmarks/benchmark_suite.py
```

## MNIST Demo

```bash
python examples/mnist_inference.py
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Matrix size | Up to 1024×1024 |
| WDM channels | 128 |
| Effective precision | 8-12 bits |
| Latency (64×64) | <100 μs |
| Energy efficiency | >10^14 OPs/J |

## Architecture

### MZI Mesh

Uses Clements decomposition to implement arbitrary NxN unitary matrices with N(N-1)/2 MZIs.

### Optical Matrix-Vector Multiply

For general matrices M = U @ S @ V^H using SVD decomposition with two MZI meshes and diagonal attenuators.

### All-Optical Nonlinearities

- **ReLU**: Saturable absorber
- **Sigmoid**: OPA gain saturation
- **Softmax**: Gain competition

## Noise Model

Simulates realistic physical effects:
- Phase setting noise (σ ~ 0.01 rad)
- Insertion loss (0.1 dB/MZI)
- Detector noise
- Thermal crosstalk

## License

MIT
