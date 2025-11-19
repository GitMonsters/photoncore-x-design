"""
PhotonCore-X 4x4 Mesh Implementation

Expands from 2x2 demo to full 4x4 with:
- 6 MZIs in Clements arrangement
- Complete unitary matrix implementation
- Auto-calibration with gradient descent
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import time


@dataclass
class MZIState:
    """State of a single MZI."""
    theta: float  # Internal phase (0 to π)
    phi: float    # External phase (0 to 2π)

    # Calibration coefficients
    theta_offset: float = 0.0
    theta_gain: float = 1.0
    phi_offset: float = 0.0
    phi_gain: float = 1.0


class Mesh4x4:
    """
    4x4 MZI mesh using Clements decomposition.

    Architecture (6 MZIs):

    Port 0 ─┬─[MZI0]─┬─────────┬─[MZI3]─┬─ Out 0
            │        │         │        │
    Port 1 ─┴────────┴─[MZI1]─┬┴────────┴─ Out 1
                              │
    Port 2 ─┬─[MZI2]─┬────────┴─[MZI4]─┬─ Out 2
            │        │                  │
    Port 3 ─┴────────┴─────────[MZI5]──┴─ Out 3

    Column 0: MZI0 (0,1), MZI2 (2,3)
    Column 1: MZI1 (1,2)
    Column 2: MZI3 (0,1), MZI4 (2,3)
    Column 3: MZI5 (1,2)
    """

    def __init__(self):
        self.n_ports = 4
        self.n_mzis = 6

        # MZI states
        self.mzis = [MZIState(0.0, 0.0) for _ in range(6)]

        # Output phases
        self.output_phases = np.zeros(4)

        # Noise parameters
        self.phase_noise_std = 0.01
        self.detector_noise_std = 0.001

    def set_phases(self, thetas, phis, output_phases):
        """Set all MZI phases."""
        thetas = list(thetas) if not isinstance(thetas, list) else thetas
        phis = list(phis) if not isinstance(phis, list) else phis

        for i in range(min(6, len(thetas))):
            self.mzis[i].theta = float(thetas[i])
            self.mzis[i].phi = float(phis[i])

        if isinstance(output_phases, np.ndarray):
            self.output_phases = output_phases.copy()
        else:
            self.output_phases = np.array(output_phases)

    def _mzi_transform(self, mzi: MZIState, add_noise: bool = False) -> np.ndarray:
        """Get 2x2 transfer matrix for single MZI."""
        theta = mzi.theta * mzi.theta_gain + mzi.theta_offset
        phi = mzi.phi * mzi.phi_gain + mzi.phi_offset

        if add_noise:
            theta += np.random.normal(0, self.phase_noise_std)
            phi += np.random.normal(0, self.phase_noise_std)

        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        exp_phi = np.exp(1j * phi)

        return np.array([
            [exp_phi * c, 1j * exp_phi * s],
            [1j * s, c]
        ], dtype=np.complex128)

    def forward(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Forward pass through 4x4 mesh using Clements structure."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'photoncore_prototype', 'src'))

        from mzi import MZIMesh

        # Create MZI mesh with our phases
        mesh = MZIMesh(4, insertion_loss_db=0.0, phase_noise_std=self.phase_noise_std if add_noise else 0.0)

        thetas = np.array([m.theta for m in self.mzis])
        phis = np.array([m.phi for m in self.mzis])

        mesh.set_phases(thetas, phis, self.output_phases)

        state = mesh.forward(x.astype(np.complex128), add_noise=add_noise)

        # Detector noise
        if add_noise:
            noise_re = np.random.normal(0, self.detector_noise_std, 4)
            noise_im = np.random.normal(0, self.detector_noise_std, 4)
            state = state + noise_re + 1j * noise_im

        return state

    def get_matrix(self, add_noise: bool = False) -> np.ndarray:
        """Get full 4x4 transfer matrix."""
        U = np.zeros((4, 4), dtype=np.complex128)

        for i in range(4):
            x = np.zeros(4, dtype=np.complex128)
            x[i] = 1.0
            U[:, i] = self.forward(x, add_noise)

        return U


def clements_decompose_4x4(U: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
    """
    Decompose 4x4 unitary matrix into MZI phases.

    Uses simplified approach - import from existing implementation.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'photoncore_prototype', 'src'))

    from clements import ClementsDecomposition

    decomp = ClementsDecomposition(4)
    thetas, phis, output_phases = decomp.decompose(U)

    return thetas, phis, output_phases


class AutoCalibrator:
    """
    Auto-calibration using gradient descent.

    Measures actual vs expected outputs and adjusts MZI calibration
    coefficients to minimize error.
    """

    def __init__(self, mesh: Mesh4x4):
        self.mesh = mesh
        self.learning_rate = 0.01
        self.n_iterations = 100

    def calibrate(self, target_matrix: np.ndarray) -> dict:
        """
        Calibrate mesh to implement target matrix.

        Uses gradient descent on calibration coefficients.
        """
        # Initial decomposition
        thetas, phis, output = clements_decompose_4x4(target_matrix)
        self.mesh.set_phases(thetas, phis, output)

        # Calibration loop
        errors = []

        for iteration in range(self.n_iterations):
            # Measure actual matrix
            actual = self.mesh.get_matrix(add_noise=True)

            # Compute error
            error = np.linalg.norm(actual - target_matrix)
            errors.append(error)

            if error < 0.01:
                break

            # Gradient estimation (finite difference)
            for i in range(6):
                # Theta offset gradient
                self.mesh.mzis[i].theta_offset += 0.001
                err_plus = np.linalg.norm(self.mesh.get_matrix(False) - target_matrix)
                self.mesh.mzis[i].theta_offset -= 0.002
                err_minus = np.linalg.norm(self.mesh.get_matrix(False) - target_matrix)
                self.mesh.mzis[i].theta_offset += 0.001

                grad = (err_plus - err_minus) / 0.002
                self.mesh.mzis[i].theta_offset -= self.learning_rate * grad

                # Phi offset gradient
                self.mesh.mzis[i].phi_offset += 0.001
                err_plus = np.linalg.norm(self.mesh.get_matrix(False) - target_matrix)
                self.mesh.mzis[i].phi_offset -= 0.002
                err_minus = np.linalg.norm(self.mesh.get_matrix(False) - target_matrix)
                self.mesh.mzis[i].phi_offset += 0.001

                grad = (err_plus - err_minus) / 0.002
                self.mesh.mzis[i].phi_offset -= self.learning_rate * grad

        return {
            'iterations': len(errors),
            'final_error': errors[-1],
            'error_history': errors,
            'calibration': [(m.theta_offset, m.phi_offset) for m in self.mesh.mzis]
        }


def demo_4x4_mesh():
    """Demonstrate 4x4 optical mesh."""
    print("=" * 60)
    print("PhotonCore-X 4x4 Mesh Demonstration")
    print("=" * 60)

    # Create mesh
    mesh = Mesh4x4()

    # Generate random unitary target
    from scipy.stats import unitary_group
    U_target = unitary_group.rvs(4)

    print("\n1. Target 4x4 Unitary Matrix:")
    for i in range(4):
        row = [f"{np.abs(U_target[i,j]):.2f}" for j in range(4)]
        print(f"   [{', '.join(row)}]")

    # Decompose
    print("\n2. Clements Decomposition:")
    thetas, phis, output = clements_decompose_4x4(U_target)

    # Convert to numpy arrays
    thetas = np.array(thetas)
    phis = np.array(phis)
    output = np.array(output)

    mesh.set_phases(thetas.tolist(), phis.tolist(), output.tolist())

    print(f"   MZI phases (θ): {[f'{t:.2f}' for t in thetas]}")
    print(f"   MZI phases (φ): {[f'{p:.2f}' for p in phis]}")

    # Verify reconstruction
    U_reconstructed = mesh.get_matrix(add_noise=False)
    recon_error = np.linalg.norm(U_reconstructed - U_target)
    print(f"\n3. Reconstruction Error: {recon_error:.6f}")

    # Run auto-calibration
    print("\n4. Auto-Calibration:")
    calibrator = AutoCalibrator(mesh)
    cal_result = calibrator.calibrate(U_target)

    print(f"   Iterations: {cal_result['iterations']}")
    print(f"   Final error: {cal_result['final_error']:.4f}")

    # Test MVM
    print("\n5. Matrix-Vector Multiply Test:")

    test_vectors = [
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, 0]),
        np.array([1, 1, 1, 1]) / 2,
    ]

    for x in test_vectors:
        y_expected = U_target @ x
        y_measured = mesh.forward(x, add_noise=True)

        error = np.linalg.norm(y_measured - y_expected)

        x_str = f"[{', '.join([f'{v:.1f}' for v in x])}]"
        print(f"   Input: {x_str}")
        print(f"   Expected |y|: [{', '.join([f'{np.abs(v):.2f}' for v in y_expected])}]")
        print(f"   Measured |y|: [{', '.join([f'{np.abs(v):.2f}' for v in y_measured])}]")
        print(f"   Error: {error:.4f}")
        print()

    # Performance metrics
    print("=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"  Matrix size: 4x4")
    print(f"  MZIs used: 6")
    print(f"  Decomposition error: {recon_error:.6f}")
    print(f"  Calibrated error: {cal_result['final_error']:.4f}")
    print(f"  Operations per MVM: {4*4*2} = 32")
    print()

    return mesh, cal_result


if __name__ == "__main__":
    demo_4x4_mesh()
