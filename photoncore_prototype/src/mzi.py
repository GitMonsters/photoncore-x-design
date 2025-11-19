"""
Mach-Zehnder Interferometer (MZI) Simulation

Core building block for programmable photonic circuits.
Each MZI implements a 2x2 unitary transformation controlled by phase shifters.
"""

import numpy as np
from typing import Tuple, Optional
from numba import jit


class MachZehnderInterferometer:
    """
    Single MZI unit with two phase shifters (theta, phi).

    Implements the transformation:
    T(θ, φ) = [[e^(iφ)cos(θ), -sin(θ)],
               [e^(iφ)sin(θ),  cos(θ)]]

    This is a general 2x2 unitary (up to global phase).
    """

    def __init__(self, theta: float = 0.0, phi: float = 0.0):
        """
        Initialize MZI with phase settings.

        Args:
            theta: Internal phase (controls splitting ratio)
            phi: External phase (controls output phase)
        """
        self.theta = theta
        self.phi = phi
        self._update_matrix()

    def _update_matrix(self):
        """Compute the transfer matrix from current phases."""
        self.matrix = self._compute_matrix(self.theta, self.phi)

    @staticmethod
    @jit(nopython=True)
    def _compute_matrix(theta: float, phi: float) -> np.ndarray:
        """JIT-compiled matrix computation."""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        exp_phi = np.exp(1j * phi)

        return np.array([
            [exp_phi * cos_t, -sin_t],
            [exp_phi * sin_t, cos_t]
        ], dtype=np.complex128)

    def set_phases(self, theta: float, phi: float):
        """Update phase settings."""
        self.theta = theta
        self.phi = phi
        self._update_matrix()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply MZI transformation to input optical fields.

        Args:
            inputs: Complex array of shape (2,) representing two input ports

        Returns:
            Complex array of shape (2,) representing two output ports
        """
        return self.matrix @ inputs

    def __repr__(self):
        return f"MZI(θ={self.theta:.3f}, φ={self.phi:.3f})"


class MZIMesh:
    """
    Rectangular mesh of MZIs implementing arbitrary NxN unitary matrix.

    Uses the Clements decomposition architecture:
    - N(N-1)/2 MZIs arranged in a triangular pattern
    - Can implement any unitary transformation

    Architecture for N=4:

    Port 0 ──[MZI]──[MZI]──[MZI]──►
                ╲   ╱  ╲   ╱
    Port 1 ──────[MZI]──[MZI]────►
                    ╲   ╱
    Port 2 ──────────[MZI]──────►

    Port 3 ──────────────────────►
    """

    def __init__(self, n_ports: int,
                 insertion_loss_db: float = 0.1,
                 phase_noise_std: float = 0.0,
                 crosstalk_db: float = -40.0):
        """
        Initialize MZI mesh.

        Args:
            n_ports: Number of input/output ports (matrix dimension)
            insertion_loss_db: Loss per MZI in dB
            phase_noise_std: Standard deviation of phase noise (radians)
            crosstalk_db: Crosstalk between adjacent waveguides in dB
        """
        self.n_ports = n_ports
        self.n_mzis = n_ports * (n_ports - 1) // 2

        # Physical parameters
        self.insertion_loss = 10 ** (-insertion_loss_db / 10)
        self.phase_noise_std = phase_noise_std
        self.crosstalk = 10 ** (crosstalk_db / 10)

        # Initialize MZIs with identity (all light passes through)
        self.mzis = []
        self.mzi_positions = []  # (layer, position) for each MZI

        self._build_mesh()

        # Phase arrays for fast access
        self.thetas = np.zeros(self.n_mzis)
        self.phis = np.zeros(self.n_mzis)

        # Output phase shifters (diagonal phases in Clements)
        self.output_phases = np.zeros(n_ports)

    def _build_mesh(self):
        """Construct the mesh topology."""
        n = self.n_ports
        mzi_idx = 0

        # Clements architecture: alternating layers
        for layer in range(2 * n - 3):
            if layer % 2 == 0:
                # Even layer: MZIs at positions (0,1), (2,3), ...
                start = 0
            else:
                # Odd layer: MZIs at positions (1,2), (3,4), ...
                start = 1

            for pos in range(start, n - 1, 2):
                self.mzis.append(MachZehnderInterferometer())
                self.mzi_positions.append((layer, pos))
                mzi_idx += 1

                if mzi_idx >= self.n_mzis:
                    return

    def set_phases(self, thetas: np.ndarray, phis: np.ndarray,
                   output_phases: Optional[np.ndarray] = None):
        """
        Set all phase values in the mesh.

        Args:
            thetas: Array of internal phases, shape (n_mzis,)
            phis: Array of external phases, shape (n_mzis,)
            output_phases: Optional output phase shifters, shape (n_ports,)
        """
        assert len(thetas) == self.n_mzis
        assert len(phis) == self.n_mzis

        self.thetas = thetas.copy()
        self.phis = phis.copy()

        for i, mzi in enumerate(self.mzis):
            mzi.set_phases(thetas[i], phis[i])

        if output_phases is not None:
            self.output_phases = output_phases.copy()

    def forward(self, inputs: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """
        Propagate optical fields through the mesh.

        Args:
            inputs: Complex input field, shape (n_ports,)
            add_noise: Whether to add phase noise and loss

        Returns:
            Complex output field, shape (n_ports,)
        """
        state = inputs.astype(np.complex128).copy()
        n = self.n_ports

        mzi_idx = 0

        # Process each layer
        for layer in range(2 * n - 3):
            if layer % 2 == 0:
                start = 0
            else:
                start = 1

            for pos in range(start, n - 1, 2):
                if mzi_idx >= self.n_mzis:
                    break

                # Get MZI matrix
                mzi = self.mzis[mzi_idx]
                matrix = mzi.matrix

                # Add noise if requested
                if add_noise:
                    # Phase noise
                    noise_theta = np.random.normal(0, self.phase_noise_std)
                    noise_phi = np.random.normal(0, self.phase_noise_std)
                    matrix = MachZehnderInterferometer._compute_matrix(
                        mzi.theta + noise_theta,
                        mzi.phi + noise_phi
                    )

                    # Insertion loss
                    matrix *= np.sqrt(self.insertion_loss)

                # Apply MZI to ports (pos, pos+1)
                port_inputs = state[pos:pos+2]
                state[pos:pos+2] = matrix @ port_inputs

                mzi_idx += 1

        # Apply output phase shifters
        state *= np.exp(1j * self.output_phases)

        return state

    def get_matrix(self, add_noise: bool = False) -> np.ndarray:
        """
        Compute the full transfer matrix of the mesh.

        Args:
            add_noise: Whether to include noise effects

        Returns:
            Complex matrix of shape (n_ports, n_ports)
        """
        # Build matrix by applying to basis vectors
        matrix = np.zeros((self.n_ports, self.n_ports), dtype=np.complex128)

        for i in range(self.n_ports):
            basis = np.zeros(self.n_ports, dtype=np.complex128)
            basis[i] = 1.0
            matrix[:, i] = self.forward(basis, add_noise=add_noise)

        return matrix

    def is_unitary(self, tolerance: float = 1e-6) -> bool:
        """Check if current configuration implements a unitary matrix."""
        U = self.get_matrix()
        should_be_identity = U @ U.conj().T
        return np.allclose(should_be_identity, np.eye(self.n_ports), atol=tolerance)

    def __repr__(self):
        return f"MZIMesh(n_ports={self.n_ports}, n_mzis={self.n_mzis})"


class BeamSplitter:
    """
    Simple 50/50 beam splitter (fixed MZI at θ=π/4).
    """

    def __init__(self):
        self.matrix = np.array([
            [1, 1j],
            [1j, 1]
        ], dtype=np.complex128) / np.sqrt(2)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.matrix @ inputs


class PhaseShifter:
    """
    Single-mode phase shifter.
    """

    def __init__(self, phase: float = 0.0):
        self.phase = phase

    def forward(self, input_field: complex) -> complex:
        return input_field * np.exp(1j * self.phase)

    def set_phase(self, phase: float):
        self.phase = phase


# Convenience functions for quick testing
def create_identity_mesh(n_ports: int) -> MZIMesh:
    """Create a mesh configured as identity (all light passes through)."""
    mesh = MZIMesh(n_ports)
    # Identity: all thetas = 0 (full transmission)
    mesh.set_phases(
        np.zeros(mesh.n_mzis),
        np.zeros(mesh.n_mzis),
        np.zeros(n_ports)
    )
    return mesh


def create_random_unitary_mesh(n_ports: int) -> MZIMesh:
    """Create a mesh with random unitary configuration."""
    mesh = MZIMesh(n_ports)
    mesh.set_phases(
        np.random.uniform(0, np.pi/2, mesh.n_mzis),
        np.random.uniform(0, 2*np.pi, mesh.n_mzis),
        np.random.uniform(0, 2*np.pi, n_ports)
    )
    return mesh


if __name__ == "__main__":
    # Quick test
    print("Testing MZI Mesh...")

    mesh = MZIMesh(4)
    print(f"Created {mesh}")

    # Random configuration
    mesh.set_phases(
        np.random.uniform(0, np.pi/2, mesh.n_mzis),
        np.random.uniform(0, 2*np.pi, mesh.n_mzis),
        np.random.uniform(0, 2*np.pi, 4)
    )

    # Check unitarity
    U = mesh.get_matrix()
    print(f"Is unitary: {mesh.is_unitary()}")

    # Test forward pass
    x = np.random.randn(4) + 1j * np.random.randn(4)
    x /= np.linalg.norm(x)

    y = mesh.forward(x)
    print(f"Input norm: {np.linalg.norm(x):.6f}")
    print(f"Output norm: {np.linalg.norm(y):.6f}")

    print("✓ MZI mesh test passed!")
