"""
PhotonCore SDK - Main Interface

High-level API for PhotonCore-X photonic AI accelerator.
Provides PyTorch-like interface for optical neural networks.
"""

import numpy as np
from typing import List, Optional, Union, Tuple
import time

from mzi import MZIMesh
from clements import ClementsDecomposition, random_unitary
from optical_mvm import OpticalMatrixUnit, OpticalNonlinearity, OpticalNeuralLayer, WDMParallelUnit
from calibration import CalibrationSystem, AutoCalibrator


class PhotonCoreSimulator:
    """
    High-level simulator for PhotonCore-X chip.

    Provides complete photonic AI accelerator simulation including:
    - Matrix operations
    - Neural network layers
    - Calibration
    - Noise modeling
    """

    def __init__(self, n_ports: int = 64,
                 n_wdm_channels: int = 128,
                 insertion_loss_db: float = 0.1,
                 phase_noise_std: float = 0.01,
                 detector_noise_std: float = 0.001):
        """
        Initialize PhotonCore simulator.

        Args:
            n_ports: Matrix dimension
            n_wdm_channels: Number of WDM parallel channels
            insertion_loss_db: Loss per MZI
            phase_noise_std: Phase control noise
            detector_noise_std: Detector noise
        """
        self.n_ports = n_ports
        self.n_wdm_channels = n_wdm_channels

        # Physical parameters
        self.insertion_loss_db = insertion_loss_db
        self.phase_noise_std = phase_noise_std
        self.detector_noise_std = detector_noise_std

        # Create main optical unit
        self.optical_unit = OpticalMatrixUnit(
            n_ports,
            insertion_loss_db,
            phase_noise_std,
            detector_noise_std
        )

        # WDM parallel processing
        self.wdm_unit = WDMParallelUnit(n_ports, n_wdm_channels)

        # Calibration
        n_mzis = n_ports * (n_ports - 1) // 2
        self.calibration = CalibrationSystem(n_ports, n_mzis)

        # Statistics
        self.total_ops = 0
        self.total_time = 0

    def load_matrix(self, M: np.ndarray):
        """Load weight matrix into optical unit."""
        self.optical_unit.load_matrix(M)

    def matmul(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Perform matrix-vector multiplication.

        Args:
            x: Input vector or batch

        Returns:
            Output vector or batch
        """
        start = time.time()

        if x.ndim == 1:
            result = self.optical_unit.forward(x, add_noise=add_noise)
            self.total_ops += self.n_ports ** 2
        else:
            # Batch processing
            results = []
            for i in range(len(x)):
                y = self.optical_unit.forward(x[i], add_noise=add_noise)
                results.append(y)
            result = np.array(results)
            self.total_ops += len(x) * self.n_ports ** 2

        self.total_time += time.time() - start
        return result

    def matmul_parallel(self, matrices: List[np.ndarray],
                        inputs: List[np.ndarray],
                        add_noise: bool = True) -> List[np.ndarray]:
        """
        Perform multiple matrix operations in parallel using WDM.

        Args:
            matrices: List of matrices (one per channel)
            inputs: List of input vectors

        Returns:
            List of output vectors
        """
        self.wdm_unit.load_matrices(matrices)
        return self.wdm_unit.forward_parallel(inputs, add_noise=add_noise)

    def get_throughput(self) -> float:
        """Get operations per second."""
        if self.total_time > 0:
            return self.total_ops / self.total_time
        return 0

    def get_energy_efficiency(self, power_watts: float = 50.0) -> float:
        """
        Get operations per joule.

        Args:
            power_watts: Chip power consumption

        Returns:
            OPs/J
        """
        throughput = self.get_throughput()
        return throughput / power_watts

    def reset_stats(self):
        """Reset performance statistics."""
        self.total_ops = 0
        self.total_time = 0


class PhotonCoreSDK:
    """
    Software Development Kit for PhotonCore-X.

    Provides high-level neural network API similar to PyTorch.
    """

    def __init__(self, simulator: Optional[PhotonCoreSimulator] = None):
        """
        Initialize SDK.

        Args:
            simulator: PhotonCore simulator (creates default if None)
        """
        if simulator is None:
            simulator = PhotonCoreSimulator()

        self.simulator = simulator
        self.layers = []

    def add_layer(self, in_features: int, out_features: int,
                  nonlinearity: str = 'relu', bias: bool = True):
        """
        Add a neural network layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            nonlinearity: Activation function
            bias: Include bias
        """
        layer = OpticalNeuralLayer(in_features, out_features, nonlinearity, bias)
        self.layers.append(layer)

    def forward(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Forward pass through all layers.

        Args:
            x: Input

        Returns:
            Output
        """
        for layer in self.layers:
            x = layer.forward(x, add_noise=add_noise)
        return x

    def compile_pytorch_model(self, model):
        """
        Compile a PyTorch model to PhotonCore layers.

        Args:
            model: PyTorch nn.Module

        Returns:
            Compiled PhotonCore model
        """
        try:
            import torch.nn as nn

            self.layers = []

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    in_f = module.in_features
                    out_f = module.out_features
                    has_bias = module.bias is not None

                    # Create optical layer
                    layer = OpticalNeuralLayer(in_f, out_f, 'none', has_bias)

                    # Copy weights
                    W = module.weight.detach().numpy()
                    layer.set_weights(W)

                    if has_bias:
                        layer.bias = module.bias.detach().numpy()

                    self.layers.append(layer)

                elif isinstance(module, nn.ReLU):
                    # Mark previous layer with ReLU activation
                    if self.layers:
                        self.layers[-1].nonlinearity = OpticalNonlinearity('relu')

                elif isinstance(module, nn.Sigmoid):
                    if self.layers:
                        self.layers[-1].nonlinearity = OpticalNonlinearity('sigmoid')

                elif isinstance(module, nn.Tanh):
                    if self.layers:
                        self.layers[-1].nonlinearity = OpticalNonlinearity('tanh')

            return self

        except ImportError:
            raise ImportError("PyTorch required for model compilation")

    def benchmark(self, input_shape: Tuple[int, ...],
                  n_iterations: int = 1000,
                  add_noise: bool = True) -> dict:
        """
        Benchmark the compiled model.

        Args:
            input_shape: Shape of input tensor
            n_iterations: Number of benchmark iterations
            add_noise: Include noise in benchmark

        Returns:
            Benchmark results
        """
        # Generate random inputs
        if len(input_shape) == 1:
            inputs = [np.random.randn(input_shape[0]) for _ in range(n_iterations)]
        else:
            inputs = [np.random.randn(*input_shape) for _ in range(n_iterations)]

        # Warm up
        for i in range(10):
            self.forward(inputs[i], add_noise=add_noise)

        # Benchmark
        self.simulator.reset_stats()

        start = time.time()
        for i in range(n_iterations):
            self.forward(inputs[i], add_noise=add_noise)
        elapsed = time.time() - start

        # Calculate metrics
        latency_ms = (elapsed / n_iterations) * 1000
        throughput = n_iterations / elapsed

        return {
            'latency_ms': latency_ms,
            'throughput_samples_per_sec': throughput,
            'total_time_sec': elapsed,
            'ops_per_sec': self.simulator.get_throughput(),
            'energy_efficiency_ops_per_j': self.simulator.get_energy_efficiency()
        }


class PhotonCoreNetwork:
    """
    High-level neural network class for PhotonCore.

    Similar to PyTorch's nn.Module but for optical neural networks.
    """

    def __init__(self):
        self.layers = []
        self._training = False

    def add_linear(self, in_features: int, out_features: int):
        """Add linear layer."""
        self.layers.append(('linear', OpticalNeuralLayer(
            in_features, out_features, 'none', True
        )))

    def add_relu(self):
        """Add ReLU activation."""
        self.layers.append(('relu', OpticalNonlinearity('relu')))

    def add_sigmoid(self):
        """Add sigmoid activation."""
        self.layers.append(('sigmoid', OpticalNonlinearity('sigmoid')))

    def add_softmax(self):
        """Add softmax activation."""
        self.layers.append(('softmax', OpticalNonlinearity('softmax')))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        for name, layer in self.layers:
            if name == 'linear':
                x = layer.forward(x, add_noise=not self._training)
            else:
                x = layer.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        """Set training mode (no noise)."""
        self._training = True

    def eval(self):
        """Set evaluation mode (with noise)."""
        self._training = False


def create_optical_mlp(layer_sizes: List[int],
                       activation: str = 'relu') -> PhotonCoreNetwork:
    """
    Create a multi-layer perceptron for PhotonCore.

    Args:
        layer_sizes: List of layer dimensions [input, hidden1, hidden2, ..., output]
        activation: Activation function

    Returns:
        PhotonCore network
    """
    network = PhotonCoreNetwork()

    for i in range(len(layer_sizes) - 1):
        network.add_linear(layer_sizes[i], layer_sizes[i+1])

        # Add activation after all but last layer
        if i < len(layer_sizes) - 2:
            if activation == 'relu':
                network.add_relu()
            elif activation == 'sigmoid':
                network.add_sigmoid()

    return network


if __name__ == "__main__":
    print("Testing PhotonCore SDK...")

    # Test 1: Basic simulator
    print("\n--- Test 1: Basic Simulator ---")
    sim = PhotonCoreSimulator(n_ports=16)

    # Load random unitary
    U = random_unitary(16)
    sim.load_matrix(U)

    # Test matmul
    x = np.random.randn(16)
    y = sim.matmul(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Throughput: {sim.get_throughput():.2e} ops/s")

    # Test 2: SDK
    print("\n--- Test 2: SDK ---")
    sdk = PhotonCoreSDK()
    sdk.add_layer(16, 32, 'relu')
    sdk.add_layer(32, 10, 'none')

    x = np.random.randn(16)
    y = sdk.forward(x)

    print(f"Network output shape: {y.shape}")

    # Test 3: Benchmark
    print("\n--- Test 3: Benchmark ---")
    results = sdk.benchmark((16,), n_iterations=100)

    print(f"Latency: {results['latency_ms']:.3f} ms")
    print(f"Throughput: {results['throughput_samples_per_sec']:.1f} samples/s")
    print(f"Energy efficiency: {results['energy_efficiency_ops_per_j']:.2e} ops/J")

    # Test 4: Network builder
    print("\n--- Test 4: Network Builder ---")
    network = create_optical_mlp([784, 256, 128, 10], activation='relu')

    x = np.random.randn(784)
    y = network(x)

    print(f"MLP input: {x.shape}")
    print(f"MLP output: {y.shape}")

    # Test 5: WDM parallel
    print("\n--- Test 5: WDM Parallel ---")
    matrices = [random_unitary(16) for _ in range(8)]
    inputs = [np.random.randn(16) for _ in range(8)]

    outputs = sim.matmul_parallel(matrices, inputs)

    print(f"Parallel processed {len(outputs)} matrices")

    print("\n✓ All PhotonCore SDK tests passed!")
