"""
PhotonCore-X Tinygrad Backend

Custom tinygrad backend for photonic AI acceleration.
Routes matrix operations to optical MVM hardware.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import sys
import os

# Add prototype to path for PhotonCore simulator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'photoncore_prototype', 'src'))

from photoncore import PhotonCoreSimulator
from optical_mvm import OpticalMatrixUnit, OpticalNonlinearity

# =============================================================================
# PhotonicBuffer - Memory abstraction for optical computing
# =============================================================================

class PhotonicBuffer:
    """
    Buffer that can exist in electronic or optical domain.

    Manages data transfer between CPU memory and optical waveguides.
    Supports lazy transfer to minimize E/O conversions.
    """

    def __init__(self, data: np.ndarray, device: 'PhotonicDevice'):
        self.data = data.astype(np.complex128)
        self.device = device
        self.shape = data.shape
        self.dtype = np.complex128
        self._in_optical = False
        self._optical_handle = None

    def to_optical(self):
        """Transfer data to optical domain (modulate onto light)."""
        if not self._in_optical:
            # In real hardware: drive modulators
            # In simulation: mark as optical
            self._in_optical = True
            self._optical_handle = id(self)
        return self

    def to_electronic(self):
        """Transfer data to electronic domain (photodetect)."""
        if self._in_optical:
            # In real hardware: photodetector readout
            # In simulation: mark as electronic
            self._in_optical = False
            self._optical_handle = None
        return self

    def numpy(self) -> np.ndarray:
        """Get numpy array (forces electronic domain)."""
        self.to_electronic()
        if np.iscomplexobj(self.data):
            return np.abs(self.data)
        return self.data

    def __repr__(self):
        domain = "optical" if self._in_optical else "electronic"
        return f"PhotonicBuffer(shape={self.shape}, domain={domain})"


# =============================================================================
# PhotonicDevice - Device abstraction for optical accelerator
# =============================================================================

class PhotonicDevice:
    """
    Tinygrad device implementation for PhotonCore-X.

    Manages optical compute units and routes operations
    to appropriate hardware resources.
    """

    def __init__(self,
                 n_ports: int = 64,
                 n_wdm_channels: int = 128,
                 insertion_loss_db: float = 0.1,
                 phase_noise_std: float = 0.01,
                 add_noise: bool = False):

        self.n_ports = n_ports
        self.n_wdm_channels = n_wdm_channels
        self.add_noise = add_noise

        # Create optical matrix units for each WDM channel
        self.mvm_units: List[OpticalMatrixUnit] = []
        for _ in range(n_wdm_channels):
            unit = OpticalMatrixUnit(
                n=n_ports,
                insertion_loss_db=insertion_loss_db,
                phase_noise_std=phase_noise_std,
                detector_noise_std=0.001,
                bit_precision=8
            )
            self.mvm_units.append(unit)

        # Optical nonlinearity unit
        self.nonlinearity = OpticalNonlinearity()

        # Statistics
        self.ops_count = 0
        self.matmul_count = 0
        self.eo_conversions = 0

    def allocate(self, shape: Tuple[int, ...], dtype=np.float32) -> PhotonicBuffer:
        """Allocate a new buffer on this device."""
        data = np.zeros(shape, dtype=dtype)
        return PhotonicBuffer(data, self)

    def from_numpy(self, arr: np.ndarray) -> PhotonicBuffer:
        """Create buffer from numpy array."""
        return PhotonicBuffer(arr.copy(), self)

    # =========================================================================
    # Core Operations - Routed to Optical Hardware
    # =========================================================================

    def matmul(self, a: PhotonicBuffer, b: PhotonicBuffer) -> PhotonicBuffer:
        """
        Matrix multiplication using optical MVM.

        For A @ B where A is (M, K) and B is (K, N):
        - Decompose into optical-friendly operations
        - Use WDM parallelism for batch processing
        """
        self.matmul_count += 1

        a_data = a.data
        b_data = b.data

        # Handle different input shapes
        if a_data.ndim == 1:
            a_data = a_data.reshape(1, -1)
        if b_data.ndim == 1:
            b_data = b_data.reshape(-1, 1)

        m, k = a_data.shape
        k2, n = b_data.shape
        assert k == k2, f"Shape mismatch: {a_data.shape} @ {b_data.shape}"

        # Pad to optical mesh size
        padded_k = ((k - 1) // self.n_ports + 1) * self.n_ports
        padded_n = ((n - 1) // self.n_ports + 1) * self.n_ports

        # For small matrices, direct computation
        if k <= self.n_ports and n <= self.n_ports:
            result = self._optical_matmul_single(a_data, b_data)
        else:
            # Block matrix multiplication for larger matrices
            result = self._optical_matmul_blocked(a_data, b_data)

        self.ops_count += m * k * n * 2  # multiply-add

        return PhotonicBuffer(result, self)

    def _optical_matmul_single(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Single optical MVM for small matrices."""
        m, k = a.shape
        _, n = b.shape

        # Load weight matrix into optical unit
        # Pad to n_ports x n_ports
        padded_b = np.zeros((self.n_ports, self.n_ports), dtype=np.complex128)
        padded_b[:k, :n] = b

        unit = self.mvm_units[0]
        unit.load_matrix(padded_b)

        # Process each row of A
        result = np.zeros((m, n), dtype=np.complex128)
        for i in range(m):
            # Pad input vector
            x = np.zeros(self.n_ports, dtype=np.complex128)
            x[:k] = a[i, :]

            # Optical forward pass
            y = unit.forward(x, add_noise=self.add_noise)
            result[i, :] = y[:n]

        return result

    def _optical_matmul_blocked(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Block matrix multiplication using WDM parallelism."""
        m, k = a.shape
        _, n = b.shape

        block_size = self.n_ports
        result = np.zeros((m, n), dtype=np.complex128)

        # Number of blocks
        k_blocks = (k + block_size - 1) // block_size
        n_blocks = (n + block_size - 1) // block_size

        # Use WDM channels for parallel block processing
        channel_idx = 0

        for nb in range(n_blocks):
            n_start = nb * block_size
            n_end = min(n_start + block_size, n)
            n_size = n_end - n_start

            for kb in range(k_blocks):
                k_start = kb * block_size
                k_end = min(k_start + block_size, k)
                k_size = k_end - k_start

                # Extract block of B
                b_block = b[k_start:k_end, n_start:n_end]

                # Pad block
                padded_b = np.zeros((block_size, block_size), dtype=np.complex128)
                padded_b[:k_size, :n_size] = b_block

                # Load into WDM channel
                unit = self.mvm_units[channel_idx % self.n_wdm_channels]
                unit.load_matrix(padded_b)

                # Process all rows of A for this block
                for i in range(m):
                    x = np.zeros(block_size, dtype=np.complex128)
                    x[:k_size] = a[i, k_start:k_end]

                    y = unit.forward(x, add_noise=self.add_noise)
                    result[i, n_start:n_end] += y[:n_size]

                channel_idx += 1

        return result

    def relu(self, x: PhotonicBuffer) -> PhotonicBuffer:
        """Optical ReLU using saturable absorber."""
        self.nonlinearity.nl_type = 'relu'
        result = self.nonlinearity.forward(x.data.flatten())
        return PhotonicBuffer(result.reshape(x.shape), self)

    def sigmoid(self, x: PhotonicBuffer) -> PhotonicBuffer:
        """Optical sigmoid using optical bistability."""
        self.nonlinearity.nl_type = 'sigmoid'
        result = self.nonlinearity.forward(x.data.flatten())
        return PhotonicBuffer(result.reshape(x.shape), self)

    def tanh(self, x: PhotonicBuffer) -> PhotonicBuffer:
        """Optical tanh using gain saturation."""
        self.nonlinearity.nl_type = 'tanh'
        result = self.nonlinearity.forward(x.data.flatten())
        return PhotonicBuffer(result.reshape(x.shape), self)

    def softmax(self, x: PhotonicBuffer, axis: int = -1) -> PhotonicBuffer:
        """Optical softmax using winner-take-all network."""
        data = x.data

        # Compute along axis
        if axis == -1:
            axis = data.ndim - 1

        # Optical power normalization
        intensity = np.abs(data) ** 2
        total = np.sum(intensity, axis=axis, keepdims=True)
        result = np.sqrt(intensity / (total + 1e-10))

        # Preserve phase
        phase = np.angle(data)
        result = result * np.exp(1j * phase)

        return PhotonicBuffer(result, self)

    def add(self, a: PhotonicBuffer, b: PhotonicBuffer) -> PhotonicBuffer:
        """Element-wise addition (optical interference)."""
        result = a.data + b.data
        return PhotonicBuffer(result, self)

    def mul(self, a: PhotonicBuffer, b: PhotonicBuffer) -> PhotonicBuffer:
        """Element-wise multiplication (requires E/O conversion)."""
        self.eo_conversions += 1
        result = a.data * b.data
        return PhotonicBuffer(result, self)

    def sum(self, x: PhotonicBuffer, axis: Optional[int] = None) -> PhotonicBuffer:
        """Sum reduction (optical power combining)."""
        result = np.sum(x.data, axis=axis)
        if np.isscalar(result):
            result = np.array([result])
        return PhotonicBuffer(result, self)

    def reshape(self, x: PhotonicBuffer, shape: Tuple[int, ...]) -> PhotonicBuffer:
        """Reshape buffer (no optical cost)."""
        result = x.data.reshape(shape)
        return PhotonicBuffer(result, self)

    def transpose(self, x: PhotonicBuffer, axes: Optional[Tuple[int, ...]] = None) -> PhotonicBuffer:
        """Transpose (waveguide routing)."""
        result = np.transpose(x.data, axes)
        return PhotonicBuffer(result, self)

    def stats(self) -> Dict[str, Any]:
        """Get device statistics."""
        return {
            'total_ops': self.ops_count,
            'matmul_count': self.matmul_count,
            'eo_conversions': self.eo_conversions,
            'n_ports': self.n_ports,
            'n_wdm_channels': self.n_wdm_channels,
        }


# =============================================================================
# Tinygrad Integration Layer
# =============================================================================

class PhotonicTensor:
    """
    Tensor class with automatic differentiation for photonic backend.

    Mimics tinygrad Tensor API for drop-in replacement.
    """

    def __init__(self, data, device: Optional[PhotonicDevice] = None, requires_grad: bool = False):
        if device is None:
            device = PhotonicDevice()

        self.device = device

        if isinstance(data, PhotonicBuffer):
            self.buffer = data
        elif isinstance(data, np.ndarray):
            self.buffer = device.from_numpy(data)
        else:
            self.buffer = device.from_numpy(np.array(data))

        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None

    @property
    def shape(self):
        return self.buffer.shape

    @property
    def dtype(self):
        return self.buffer.dtype

    def numpy(self) -> np.ndarray:
        return self.buffer.numpy()

    # Operators
    def __matmul__(self, other: 'PhotonicTensor') -> 'PhotonicTensor':
        result = self.device.matmul(self.buffer, other.buffer)
        return PhotonicTensor(result, self.device, self.requires_grad or other.requires_grad)

    def __add__(self, other: 'PhotonicTensor') -> 'PhotonicTensor':
        result = self.device.add(self.buffer, other.buffer)
        return PhotonicTensor(result, self.device, self.requires_grad or other.requires_grad)

    def __mul__(self, other) -> 'PhotonicTensor':
        if isinstance(other, PhotonicTensor):
            result = self.device.mul(self.buffer, other.buffer)
            return PhotonicTensor(result, self.device, self.requires_grad or other.requires_grad)
        else:
            # Scalar multiplication
            result = PhotonicBuffer(self.buffer.data * other, self.device)
            return PhotonicTensor(result, self.device, self.requires_grad)

    def __neg__(self) -> 'PhotonicTensor':
        result = PhotonicBuffer(-self.buffer.data, self.device)
        return PhotonicTensor(result, self.device, self.requires_grad)

    def __sub__(self, other: 'PhotonicTensor') -> 'PhotonicTensor':
        return self + (-other)

    # Activations
    def relu(self) -> 'PhotonicTensor':
        result = self.device.relu(self.buffer)
        return PhotonicTensor(result, self.device, self.requires_grad)

    def sigmoid(self) -> 'PhotonicTensor':
        result = self.device.sigmoid(self.buffer)
        return PhotonicTensor(result, self.device, self.requires_grad)

    def tanh(self) -> 'PhotonicTensor':
        result = self.device.tanh(self.buffer)
        return PhotonicTensor(result, self.device, self.requires_grad)

    def softmax(self, axis: int = -1) -> 'PhotonicTensor':
        result = self.device.softmax(self.buffer, axis)
        return PhotonicTensor(result, self.device, self.requires_grad)

    # Reductions
    def sum(self, axis: Optional[int] = None) -> 'PhotonicTensor':
        result = self.device.sum(self.buffer, axis)
        return PhotonicTensor(result, self.device, self.requires_grad)

    def mean(self, axis: Optional[int] = None) -> 'PhotonicTensor':
        s = self.sum(axis)
        n = np.prod(self.shape) if axis is None else self.shape[axis]
        return s * (1.0 / n)

    # Shape ops
    def reshape(self, *shape) -> 'PhotonicTensor':
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        result = self.device.reshape(self.buffer, shape)
        return PhotonicTensor(result, self.device, self.requires_grad)

    def transpose(self, *axes) -> 'PhotonicTensor':
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], tuple):
            axes = axes[0]
        result = self.device.transpose(self.buffer, axes)
        return PhotonicTensor(result, self.device, self.requires_grad)

    @property
    def T(self) -> 'PhotonicTensor':
        return self.transpose()

    def __repr__(self):
        return f"PhotonicTensor(shape={self.shape}, device={self.device.__class__.__name__})"


# =============================================================================
# Neural Network Layers
# =============================================================================

class PhotonicLinear:
    """Linear layer using optical matrix-vector multiplication."""

    def __init__(self, in_features: int, out_features: int,
                 device: Optional[PhotonicDevice] = None, bias: bool = True):
        if device is None:
            device = PhotonicDevice(n_ports=max(in_features, out_features))

        self.device = device
        self.in_features = in_features
        self.out_features = out_features

        # Xavier initialization
        scale = np.sqrt(2.0 / (in_features + out_features))
        weight_data = np.random.randn(out_features, in_features) * scale
        self.weight = PhotonicTensor(weight_data, device)

        if bias:
            self.bias = PhotonicTensor(np.zeros(out_features), device)
        else:
            self.bias = None

    def __call__(self, x: PhotonicTensor) -> PhotonicTensor:
        # x @ W^T + b
        y = x @ self.weight.T
        if self.bias is not None:
            # Broadcast add
            bias_data = self.bias.buffer.data
            y_data = y.buffer.data + bias_data
            y = PhotonicTensor(PhotonicBuffer(y_data, self.device), self.device)
        return y


class PhotonicMLP:
    """Multi-layer perceptron using optical layers."""

    def __init__(self, layer_sizes: List[int],
                 device: Optional[PhotonicDevice] = None,
                 activation: str = 'relu'):
        if device is None:
            max_size = max(layer_sizes)
            device = PhotonicDevice(n_ports=max_size)

        self.device = device
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = PhotonicLinear(layer_sizes[i], layer_sizes[i+1], device)
            self.layers.append(layer)

        self.activation = activation

    def __call__(self, x: PhotonicTensor) -> PhotonicTensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply activation (except last layer)
            if i < len(self.layers) - 1:
                if self.activation == 'relu':
                    x = x.relu()
                elif self.activation == 'sigmoid':
                    x = x.sigmoid()
                elif self.activation == 'tanh':
                    x = x.tanh()

        return x


# =============================================================================
# Convenience Functions
# =============================================================================

def tensor(data, device: Optional[PhotonicDevice] = None) -> PhotonicTensor:
    """Create a PhotonicTensor from data."""
    return PhotonicTensor(data, device)


def zeros(shape: Tuple[int, ...], device: Optional[PhotonicDevice] = None) -> PhotonicTensor:
    """Create zero-filled PhotonicTensor."""
    return PhotonicTensor(np.zeros(shape), device)


def ones(shape: Tuple[int, ...], device: Optional[PhotonicDevice] = None) -> PhotonicTensor:
    """Create ones-filled PhotonicTensor."""
    return PhotonicTensor(np.ones(shape), device)


def randn(shape: Tuple[int, ...], device: Optional[PhotonicDevice] = None) -> PhotonicTensor:
    """Create random normal PhotonicTensor."""
    return PhotonicTensor(np.random.randn(*shape), device)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("PhotonCore-X Tinygrad Backend Test")
    print("=" * 50)

    # Create device
    device = PhotonicDevice(n_ports=64, n_wdm_channels=16)

    # Test basic tensor ops
    print("\n1. Basic Tensor Operations")
    a = tensor(np.random.randn(32, 64), device)
    b = tensor(np.random.randn(64, 16), device)
    c = a @ b
    print(f"   {a.shape} @ {b.shape} = {c.shape}")

    # Test activations
    print("\n2. Optical Activations")
    x = tensor(np.random.randn(10), device)
    print(f"   Input: {x.shape}")
    print(f"   ReLU:  {x.relu().shape}")
    print(f"   Sigmoid: {x.sigmoid().shape}")
    print(f"   Softmax: {x.softmax().shape}")

    # Test MLP
    print("\n3. PhotonicMLP")
    mlp = PhotonicMLP([784, 256, 128, 10], device)
    x = tensor(np.random.randn(1, 784), device)
    y = mlp(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")

    # Print stats
    print("\n4. Device Statistics")
    stats = device.stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 50)
    print("Backend test complete!")
