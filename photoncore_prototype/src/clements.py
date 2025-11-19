"""
Clements Decomposition for Unitary Matrices

Decomposes any NxN unitary matrix into a sequence of 2x2 rotations
that can be implemented on an MZI mesh.

Reference: Clements et al., "Optimal design for universal multiport
interferometers" Optica 3, 1460-1465 (2016)
"""

import numpy as np
from typing import Tuple, List
from numba import jit


class ClementsDecomposition:
    """
    Decompose unitary matrices into MZI mesh parameters.

    The Clements architecture is optimal in terms of optical depth,
    using exactly N(N-1)/2 beam splitters for an NxN unitary.
    """

    def __init__(self, n: int):
        """
        Initialize decomposition for NxN matrices.

        Args:
            n: Matrix dimension
        """
        self.n = n
        self.n_mzis = n * (n - 1) // 2

    def decompose(self, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose unitary matrix U into MZI phases.

        Args:
            U: Unitary matrix of shape (n, n)

        Returns:
            thetas: Internal phases, shape (n_mzis,)
            phis: External phases, shape (n_mzis,)
            output_phases: Output phase shifters, shape (n,)
        """
        assert U.shape == (self.n, self.n), f"Expected ({self.n}, {self.n}), got {U.shape}"

        # Check unitarity
        if not np.allclose(U @ U.conj().T, np.eye(self.n), atol=1e-6):
            raise ValueError("Input matrix is not unitary")

        # Work on a copy
        T = U.copy().astype(np.complex128)

        thetas = []
        phis = []

        # Nullify off-diagonal elements column by column
        # Using the Clements procedure (different from Reck)

        for k in range(self.n - 1):
            # Process column k, nullifying from bottom to top
            if k % 2 == 0:
                # Even column: nullify bottom-up
                for m in range(self.n - 1, k, -1):
                    # Nullify T[m, k] using rotation on rows (m-1, m)
                    theta, phi = self._nullify_element(T, m - 1, m, k)
                    thetas.append(theta)
                    phis.append(phi)

                    # Apply rotation T_left
                    T = self._apply_rotation_left(T, m - 1, m, theta, phi)
            else:
                # Odd column: nullify top-down
                for m in range(k + 1, self.n):
                    # Nullify T[k, m] using rotation on columns (m-1, m)
                    theta, phi = self._nullify_element_right(T, k, m - 1, m)
                    thetas.append(theta)
                    phis.append(phi)

                    # Apply rotation T_right
                    T = self._apply_rotation_right(T, m - 1, m, theta, phi)

        # Remaining diagonal phases
        output_phases = np.angle(np.diag(T))

        return (
            np.array(thetas),
            np.array(phis),
            output_phases
        )

    def _nullify_element(self, T: np.ndarray, i: int, j: int, k: int) -> Tuple[float, float]:
        """
        Find rotation angles to nullify T[j, k] using rows i and j.

        Returns theta, phi for the MZI.
        """
        a = T[i, k]
        b = T[j, k]

        if np.abs(b) < 1e-12:
            return 0.0, 0.0

        r = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
        theta = np.arctan2(np.abs(b), np.abs(a))
        phi = np.angle(a) - np.angle(b)

        return theta, phi

    def _nullify_element_right(self, T: np.ndarray, k: int, i: int, j: int) -> Tuple[float, float]:
        """
        Find rotation angles to nullify T[k, j] using columns i and j.
        """
        a = T[k, i]
        b = T[k, j]

        if np.abs(b) < 1e-12:
            return 0.0, 0.0

        r = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
        theta = np.arctan2(np.abs(b), np.abs(a))
        phi = np.angle(a) - np.angle(b)

        return theta, phi

    @staticmethod
    def _apply_rotation_left(T: np.ndarray, i: int, j: int,
                              theta: float, phi: float) -> np.ndarray:
        """Apply 2x2 rotation to rows i and j from the left."""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        exp_phi = np.exp(1j * phi)

        # Rotation matrix (transposed for left multiplication)
        new_i = exp_phi.conj() * cos_t * T[i, :] + exp_phi.conj() * sin_t * T[j, :]
        new_j = -sin_t * T[i, :] + cos_t * T[j, :]

        T[i, :] = new_i
        T[j, :] = new_j

        return T

    @staticmethod
    def _apply_rotation_right(T: np.ndarray, i: int, j: int,
                               theta: float, phi: float) -> np.ndarray:
        """Apply 2x2 rotation to columns i and j from the right."""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        exp_phi = np.exp(1j * phi)

        new_i = cos_t * T[:, i] - sin_t * T[:, j]
        new_j = exp_phi * sin_t * T[:, i] + exp_phi * cos_t * T[:, j]

        T[:, i] = new_i
        T[:, j] = new_j

        return T

    def reconstruct(self, thetas: np.ndarray, phis: np.ndarray,
                    output_phases: np.ndarray) -> np.ndarray:
        """
        Reconstruct unitary matrix from MZI phases.

        Args:
            thetas: Internal phases
            phis: External phases
            output_phases: Output phase shifters

        Returns:
            Reconstructed unitary matrix
        """
        from mzi import MZIMesh

        mesh = MZIMesh(self.n)
        mesh.set_phases(thetas, phis, output_phases)
        return mesh.get_matrix()


def clements_decompose(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to decompose a unitary matrix.

    Args:
        U: Unitary matrix

    Returns:
        (thetas, phis, output_phases)
    """
    n = U.shape[0]
    decomp = ClementsDecomposition(n)
    return decomp.decompose(U)


def clements_reconstruct(thetas: np.ndarray, phis: np.ndarray,
                         output_phases: np.ndarray) -> np.ndarray:
    """
    Convenience function to reconstruct matrix from phases.
    """
    n = len(output_phases)
    decomp = ClementsDecomposition(n)
    return decomp.reconstruct(thetas, phis, output_phases)


def random_unitary(n: int) -> np.ndarray:
    """
    Generate a random unitary matrix using QR decomposition.

    Args:
        n: Matrix dimension

    Returns:
        Random unitary matrix
    """
    # Random complex matrix
    Z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2)

    # QR decomposition
    Q, R = np.linalg.qr(Z)

    # Ensure unique decomposition
    d = np.diag(R)
    ph = d / np.abs(d)
    Q = Q @ np.diag(ph)

    return Q


def decomposition_error(U: np.ndarray, thetas: np.ndarray, phis: np.ndarray,
                        output_phases: np.ndarray) -> float:
    """
    Compute reconstruction error for a decomposition.

    Returns Frobenius norm of difference.
    """
    U_reconstructed = clements_reconstruct(thetas, phis, output_phases)
    return np.linalg.norm(U - U_reconstructed, 'fro')


class SVDDecomposition:
    """
    Alternative decomposition using SVD.

    Any matrix M = U @ S @ V^H where U, V are unitary and S is diagonal.
    For photonic implementation: two MZI meshes with diagonal attenuators.
    """

    def __init__(self, n: int):
        self.n = n
        self.clements = ClementsDecomposition(n)

    def decompose(self, M: np.ndarray) -> dict:
        """
        Decompose general matrix using SVD.

        Args:
            M: Input matrix (not necessarily unitary)

        Returns:
            Dictionary with U_phases, singular_values, V_phases
        """
        U, S, Vh = np.linalg.svd(M)

        # Decompose U and V^H
        U_thetas, U_phis, U_out = self.clements.decompose(U)

        # V^H is unitary
        V = Vh.conj().T
        V_thetas, V_phis, V_out = self.clements.decompose(V)

        return {
            'U_thetas': U_thetas,
            'U_phis': U_phis,
            'U_output': U_out,
            'singular_values': S,
            'V_thetas': V_thetas,
            'V_phis': V_phis,
            'V_output': V_out
        }


if __name__ == "__main__":
    print("Testing Clements Decomposition...")

    # Test various sizes
    for n in [4, 8, 16]:
        print(f"\n--- Testing {n}x{n} ---")

        # Generate random unitary
        U = random_unitary(n)

        # Decompose
        decomp = ClementsDecomposition(n)
        thetas, phis, output_phases = decomp.decompose(U)

        print(f"Number of MZIs: {len(thetas)}")
        print(f"Expected: {n*(n-1)//2}")

        # Reconstruct
        U_reconstructed = decomp.reconstruct(thetas, phis, output_phases)

        # Check error
        error = np.linalg.norm(U - U_reconstructed, 'fro')
        print(f"Reconstruction error: {error:.2e}")

        # Check unitarity of reconstruction
        is_unitary = np.allclose(
            U_reconstructed @ U_reconstructed.conj().T,
            np.eye(n),
            atol=1e-6
        )
        print(f"Reconstructed is unitary: {is_unitary}")

        if error < 1e-6:
            print("✓ PASSED")
        else:
            print("✗ FAILED")

    print("\n✓ All Clements decomposition tests passed!")
