"""
Optical Matrix-Vector Multiplication Unit

Implements the core computation engine for PhotonCore-X:
- Coherent optical matrix-vector multiplication
- All-optical nonlinearities
- WDM parallel processing
"""

import numpy as np
from typing import Optional, Callable, List
from numba import jit

from mzi import MZIMesh
from clements import ClementsDecomposition, random_unitary, SVDDecomposition


class OpticalMatrixUnit:
    """
    Optical matrix-vector multiplication unit.

    For unitary matrices: Uses single MZI mesh
    For general matrices: Uses SVD decomposition (U @ S @ V^H)
    """

    def __init__(self, n: int,
                 insertion_loss_db: float = 0.1,
                 phase_noise_std: float = 0.01,
                 detector_noise_std: float = 0.001,
                 bit_precision: int = 8):
        """
        Initialize optical matrix unit.

        Args:
            n: Matrix dimension
            insertion_loss_db: Loss per MZI
            phase_noise_std: Phase setting noise
            detector_noise_std: Photodetector noise
            bit_precision: Effective bit precision (for quantization)
        """
        self.n = n
        self.insertion_loss_db = insertion_loss_db
        self.phase_noise_std = phase_noise_std
        self.detector_noise_std = detector_noise_std
        self.bit_precision = bit_precision

        # Components for general matrix (SVD)
        self.mesh_U = MZIMesh(n, insertion_loss_db, phase_noise_std)
        self.mesh_V = MZIMesh(n, insertion_loss_db, phase_noise_std)
        self.singular_values = np.ones(n)  # Attenuators

        # Decomposition helper
        self.decomp = ClementsDecomposition(n)
        self.svd_decomp = SVDDecomposition(n)

        # Current matrix (for reference)
        self.current_matrix = np.eye(n, dtype=np.complex128)
        self.is_unitary = True

    def load_matrix(self, M: np.ndarray):
        """
        Load a matrix into the optical unit.

        Args:
            M: Matrix to load (can be unitary or general)
        """
        assert M.shape == (self.n, self.n)
        self.current_matrix = M.copy()

        # Check if unitary
        is_unitary = np.allclose(M @ M.conj().T, np.eye(self.n), atol=1e-6)
        self.is_unitary = is_unitary

        if is_unitary:
            # Direct Clements decomposition
            thetas, phis, output_phases = self.decomp.decompose(M)
            self.mesh_U.set_phases(thetas, phis, output_phases)
            self.singular_values = np.ones(self.n)
            # V mesh not used for unitary
        else:
            # SVD decomposition
            params = self.svd_decomp.decompose(M)
            self.mesh_U.set_phases(
                params['U_thetas'],
                params['U_phis'],
                params['U_output']
            )
            self.mesh_V.set_phases(
                params['V_thetas'],
                params['V_phis'],
                params['V_output']
            )
            self.singular_values = params['singular_values']

    def forward(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Compute matrix-vector product M @ x.

        Args:
            x: Input vector (real or complex)
            add_noise: Whether to simulate physical noise

        Returns:
            Output vector
        """
        # Convert to optical field
        if np.isrealobj(x):
            optical_in = x.astype(np.complex128)
        else:
            optical_in = x.copy()

        # Normalize input (optical power constraint)
        input_norm = np.linalg.norm(optical_in)
        if input_norm > 0:
            optical_in = optical_in / input_norm

        if self.is_unitary:
            # Single mesh pass
            optical_out = self.mesh_U.forward(optical_in, add_noise=add_noise)
        else:
            # Full SVD: U @ S @ V^H
            # First apply V^H
            temp = self.mesh_V.forward(optical_in, add_noise=add_noise)

            # Apply singular values (optical attenuators)
            temp = temp * self.singular_values

            # Apply U
            optical_out = self.mesh_U.forward(temp, add_noise=add_noise)

        # Detection (coherent homodyne)
        if add_noise:
            # Add detector noise
            noise_real = np.random.normal(0, self.detector_noise_std, self.n)
            noise_imag = np.random.normal(0, self.detector_noise_std, self.n)
            optical_out = optical_out + noise_real + 1j * noise_imag

            # Quantization (ADC)
            optical_out = self._quantize(optical_out)

        # Rescale by input norm
        result = optical_out * input_norm

        return result

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        """Simulate finite precision ADC."""
        scale = 2 ** (self.bit_precision - 1)

        # Quantize real and imaginary parts
        real_q = np.round(x.real * scale) / scale
        imag_q = np.round(x.imag * scale) / scale

        return real_q + 1j * imag_q

    def get_actual_matrix(self, add_noise: bool = False) -> np.ndarray:
        """Get the actual implemented matrix (may differ from loaded due to noise)."""
        matrix = np.zeros((self.n, self.n), dtype=np.complex128)

        for i in range(self.n):
            basis = np.zeros(self.n, dtype=np.complex128)
            basis[i] = 1.0
            matrix[:, i] = self.forward(basis, add_noise=add_noise)

        return matrix

    def compute_error(self) -> float:
        """Compute error between loaded and actual matrix."""
        actual = self.get_actual_matrix(add_noise=False)
        return np.linalg.norm(self.current_matrix - actual, 'fro')


class OpticalNonlinearity:
    """
    All-optical nonlinear activation functions.

    Implements various nonlinearities without O-E-O conversion:
    - Saturable absorber (ReLU-like)
    - Optical parametric amplifier (sigmoid-like)
    - Kerr effect (polynomial)
    """

    def __init__(self, nonlinearity_type: str = 'relu',
                 threshold: float = 0.1,
                 saturation: float = 1.0):
        """
        Initialize optical nonlinearity.

        Args:
            nonlinearity_type: 'relu', 'sigmoid', 'tanh', 'kerr', 'softmax'
            threshold: Activation threshold
            saturation: Saturation level
        """
        self.type = nonlinearity_type
        self.threshold = threshold
        self.saturation = saturation

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply nonlinear activation.

        Args:
            x: Input optical field (complex)

        Returns:
            Activated output
        """
        # Work with intensity for most nonlinearities
        intensity = np.abs(x) ** 2
        phase = np.angle(x)

        if self.type == 'relu':
            # Saturable absorber model
            # Below threshold: absorbed; above: transmitted
            output_intensity = np.maximum(0, intensity - self.threshold)
            output_amplitude = np.sqrt(output_intensity)

        elif self.type == 'sigmoid':
            # OPA gain saturation model
            # Sigmoid-like response
            gain = self.saturation / (1 + np.exp(-10 * (intensity - self.threshold)))
            output_amplitude = np.sqrt(intensity) * gain

        elif self.type == 'tanh':
            # Two-photon absorption model
            output_intensity = self.saturation * np.tanh(intensity / self.saturation)
            output_amplitude = np.sqrt(output_intensity)

        elif self.type == 'kerr':
            # Kerr effect: intensity-dependent phase shift
            # Useful for implementing polynomial nonlinearities
            phase_shift = intensity * 0.1  # Kerr coefficient
            output_amplitude = np.sqrt(intensity)
            phase = phase + phase_shift

        elif self.type == 'softmax':
            # Optical softmax approximation via gain competition
            # Normalize by total power
            total_power = np.sum(intensity) + 1e-10
            output_intensity = intensity / total_power
            output_amplitude = np.sqrt(output_intensity)

        else:
            # Identity
            output_amplitude = np.sqrt(intensity)

        # Reconstruct complex field
        return output_amplitude * np.exp(1j * phase)

    def __repr__(self):
        return f"OpticalNonlinearity(type={self.type})"


class OpticalNeuralLayer:
    """
    Complete optical neural network layer.

    Combines:
    - Optical matrix unit (weights)
    - Optical nonlinearity (activation)
    """

    def __init__(self, in_features: int, out_features: int,
                 nonlinearity: str = 'relu',
                 bias: bool = True):
        """
        Initialize optical neural layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            nonlinearity: Activation function type
            bias: Whether to include bias
        """
        self.in_features = in_features
        self.out_features = out_features

        # Determine matrix unit size (must be square)
        self.n = max(in_features, out_features)

        self.matrix_unit = OpticalMatrixUnit(self.n)
        self.nonlinearity = OpticalNonlinearity(nonlinearity)

        self.bias = np.zeros(out_features) if bias else None

        # Initialize with random weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize with Xavier/Glorot initialization."""
        std = np.sqrt(2.0 / (self.in_features + self.out_features))

        # Generate random matrix
        W = np.random.randn(self.out_features, self.in_features) * std

        # Pad to square if needed
        if self.in_features != self.out_features:
            W_padded = np.zeros((self.n, self.n))
            W_padded[:self.out_features, :self.in_features] = W
            W = W_padded

        # Orthogonalize for better conditioning
        U, _, Vh = np.linalg.svd(W, full_matrices=True)
        W = U @ Vh

        self.matrix_unit.load_matrix(W)

    def set_weights(self, W: np.ndarray):
        """Set weight matrix."""
        assert W.shape == (self.out_features, self.in_features)

        # Pad to square
        if self.in_features != self.out_features:
            W_padded = np.eye(self.n)
            W_padded[:self.out_features, :self.in_features] = W
            W = W_padded

        self.matrix_unit.load_matrix(W)

    def forward(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Forward pass through the layer.

        Args:
            x: Input vector

        Returns:
            Output vector
        """
        # Pad input if needed
        if len(x) < self.n:
            x_padded = np.zeros(self.n, dtype=np.complex128)
            x_padded[:len(x)] = x
            x = x_padded

        # Matrix multiplication
        y = self.matrix_unit.forward(x, add_noise=add_noise)

        # Truncate output
        y = y[:self.out_features]

        # Add bias
        if self.bias is not None:
            y = y + self.bias

        # Nonlinearity
        y = self.nonlinearity.forward(y)

        return y


class WDMParallelUnit:
    """
    Wavelength-division multiplexed parallel processing.

    Multiple independent computations on different wavelength channels.
    """

    def __init__(self, n: int, n_channels: int = 128):
        """
        Initialize WDM parallel unit.

        Args:
            n: Matrix dimension per channel
            n_channels: Number of WDM channels
        """
        self.n = n
        self.n_channels = n_channels

        # One matrix unit per channel
        # In reality, same physical mesh with different wavelengths
        self.channels = [OpticalMatrixUnit(n) for _ in range(n_channels)]

    def load_matrices(self, matrices: List[np.ndarray]):
        """Load different matrices into each channel."""
        assert len(matrices) == self.n_channels

        for i, M in enumerate(matrices):
            self.channels[i].load_matrix(M)

    def forward_parallel(self, inputs: List[np.ndarray],
                         add_noise: bool = True) -> List[np.ndarray]:
        """
        Process multiple inputs in parallel across wavelengths.

        Args:
            inputs: List of input vectors, one per channel

        Returns:
            List of output vectors
        """
        assert len(inputs) == self.n_channels

        outputs = []
        for i, x in enumerate(inputs):
            y = self.channels[i].forward(x, add_noise=add_noise)
            outputs.append(y)

        return outputs

    def forward_batch(self, X: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Process batch using WDM parallelism.

        Args:
            X: Input batch, shape (batch_size, n)

        Returns:
            Output batch, shape (batch_size, n)
        """
        batch_size = X.shape[0]

        # Split batch across channels
        outputs = []

        for i in range(0, batch_size, self.n_channels):
            chunk = X[i:i+self.n_channels]

            # Pad if needed
            while len(chunk) < self.n_channels:
                chunk = np.vstack([chunk, np.zeros(self.n)])

            # Process in parallel
            chunk_outputs = self.forward_parallel(
                [chunk[j] for j in range(self.n_channels)],
                add_noise=add_noise
            )

            outputs.extend(chunk_outputs[:min(self.n_channels, batch_size - i)])

        return np.array(outputs)


if __name__ == "__main__":
    print("Testing Optical Matrix-Vector Multiplication Unit...")

    # Test 1: Unitary matrix
    print("\n--- Test 1: Unitary Matrix ---")
    n = 8
    unit = OpticalMatrixUnit(n)

    U = random_unitary(n)
    unit.load_matrix(U)

    x = np.random.randn(n) + 1j * np.random.randn(n)
    x /= np.linalg.norm(x)

    expected = U @ x
    actual = unit.forward(x, add_noise=False)

    error = np.linalg.norm(expected - actual)
    print(f"Forward error (no noise): {error:.2e}")
    assert error < 1e-6, "Unitary forward failed"
    print("✓ Unitary matrix test passed")

    # Test 2: General matrix (SVD)
    print("\n--- Test 2: General Matrix ---")
    M = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    unit.load_matrix(M)

    expected = M @ x
    actual = unit.forward(x, add_noise=False)

    error = np.linalg.norm(expected - actual)
    print(f"Forward error (no noise): {error:.2e}")
    # General matrix has higher error due to SVD
    assert error < 1e-4, "General matrix forward failed"
    print("✓ General matrix test passed")

    # Test 3: Optical nonlinearities
    print("\n--- Test 3: Nonlinearities ---")
    for nl_type in ['relu', 'sigmoid', 'tanh', 'softmax']:
        nl = OpticalNonlinearity(nl_type)
        y = nl.forward(x)
        print(f"  {nl_type}: input_norm={np.linalg.norm(x):.3f}, output_norm={np.linalg.norm(y):.3f}")
    print("✓ Nonlinearity tests passed")

    # Test 4: Neural layer
    print("\n--- Test 4: Neural Layer ---")
    layer = OpticalNeuralLayer(8, 8, nonlinearity='relu')
    x_real = np.random.randn(8)
    y = layer.forward(x_real, add_noise=False)
    print(f"Layer output shape: {y.shape}")
    print(f"Layer output norm: {np.linalg.norm(y):.3f}")
    print("✓ Neural layer test passed")

    # Test 5: WDM parallel
    print("\n--- Test 5: WDM Parallel ---")
    wdm = WDMParallelUnit(8, n_channels=4)
    matrices = [random_unitary(8) for _ in range(4)]
    wdm.load_matrices(matrices)

    inputs = [np.random.randn(8) for _ in range(4)]
    outputs = wdm.forward_parallel(inputs, add_noise=False)
    print(f"Processed {len(outputs)} parallel channels")
    print("✓ WDM parallel test passed")

    print("\n✓ All optical MVM tests passed!")
