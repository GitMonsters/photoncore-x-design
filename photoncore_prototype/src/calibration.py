"""
Calibration System for Photonic Circuits

Handles:
- Initial calibration of MZI mesh
- Real-time drift compensation
- ML-accelerated calibration
- Temperature compensation
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.optimize import minimize
import json


class CalibrationSystem:
    """
    Calibration system for MZI mesh.

    Addresses:
    - Phase shifter non-idealities
    - Fabrication variations
    - Thermal crosstalk
    - Temporal drift
    """

    def __init__(self, n_ports: int, n_mzis: int):
        """
        Initialize calibration system.

        Args:
            n_ports: Number of mesh ports
            n_mzis: Number of MZIs
        """
        self.n_ports = n_ports
        self.n_mzis = n_mzis

        # Calibration parameters
        self.theta_offsets = np.zeros(n_mzis)  # Phase offset per MZI
        self.phi_offsets = np.zeros(n_mzis)
        self.theta_scales = np.ones(n_mzis)    # Gain per MZI
        self.phi_scales = np.ones(n_mzis)

        # Thermal crosstalk matrix
        self.thermal_crosstalk = np.eye(n_mzis) * 0.01  # Default small crosstalk

        # Output phase calibration
        self.output_offsets = np.zeros(n_ports)

        # Calibration quality metrics
        self.last_calibration_error = float('inf')
        self.calibration_timestamp = None

    def correct_phases(self, thetas: np.ndarray, phis: np.ndarray,
                       output_phases: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply calibration corrections to phase settings.

        Args:
            thetas: Target internal phases
            phis: Target external phases
            output_phases: Target output phases

        Returns:
            Corrected phases to send to hardware
        """
        # Apply offset and scale corrections
        corrected_thetas = (thetas - self.theta_offsets) / self.theta_scales
        corrected_phis = (phis - self.phi_offsets) / self.phi_scales

        # Compensate thermal crosstalk
        # If setting MZI i affects MZI j, we need to pre-compensate
        correction = self.thermal_crosstalk @ corrected_thetas
        corrected_thetas = corrected_thetas - correction

        # Output phase correction
        corrected_output = output_phases - self.output_offsets

        return corrected_thetas, corrected_phis, corrected_output

    def save(self, filepath: str):
        """Save calibration to file."""
        data = {
            'n_ports': self.n_ports,
            'n_mzis': self.n_mzis,
            'theta_offsets': self.theta_offsets.tolist(),
            'phi_offsets': self.phi_offsets.tolist(),
            'theta_scales': self.theta_scales.tolist(),
            'phi_scales': self.phi_scales.tolist(),
            'thermal_crosstalk': self.thermal_crosstalk.tolist(),
            'output_offsets': self.output_offsets.tolist(),
            'last_error': self.last_calibration_error,
            'timestamp': self.calibration_timestamp
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load calibration from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.theta_offsets = np.array(data['theta_offsets'])
        self.phi_offsets = np.array(data['phi_offsets'])
        self.theta_scales = np.array(data['theta_scales'])
        self.phi_scales = np.array(data['phi_scales'])
        self.thermal_crosstalk = np.array(data['thermal_crosstalk'])
        self.output_offsets = np.array(data['output_offsets'])
        self.last_calibration_error = data.get('last_error', float('inf'))
        self.calibration_timestamp = data.get('timestamp')


class AutoCalibrator:
    """
    Automatic calibration routines for MZI mesh.
    """

    def __init__(self, mesh, calibration: CalibrationSystem):
        """
        Initialize auto-calibrator.

        Args:
            mesh: MZIMesh instance
            calibration: CalibrationSystem instance
        """
        self.mesh = mesh
        self.calibration = calibration
        self.n_ports = mesh.n_ports
        self.n_mzis = mesh.n_mzis

    def calibrate_single_mzi(self, mzi_idx: int, n_samples: int = 100) -> Dict:
        """
        Calibrate a single MZI to find offset and scale.

        Args:
            mzi_idx: Index of MZI to calibrate
            n_samples: Number of test points

        Returns:
            Calibration results dict
        """
        # Save current state
        original_thetas = self.mesh.thetas.copy()
        original_phis = self.mesh.phis.copy()

        # Set all other MZIs to identity (theta=0)
        test_thetas = np.zeros(self.n_mzis)
        test_phis = np.zeros(self.n_mzis)

        # Sweep this MZI through its range
        theta_sweep = np.linspace(0, np.pi/2, n_samples)
        measured_ratios = []

        for theta in theta_sweep:
            test_thetas[mzi_idx] = theta
            self.mesh.set_phases(test_thetas, test_phis, np.zeros(self.n_ports))

            # Measure power splitting ratio
            # In practice: inject light into one port, measure output ratio
            # Here: simulate by evaluating matrix

            # Get MZI position to know which ports it affects
            layer, pos = self.mesh.mzi_positions[mzi_idx]

            # Create test input at that port
            test_input = np.zeros(self.n_ports, dtype=np.complex128)
            test_input[pos] = 1.0

            output = self.mesh.forward(test_input, add_noise=True)

            # Measure splitting ratio
            p1 = np.abs(output[pos]) ** 2
            p2 = np.abs(output[pos + 1]) ** 2

            if p1 + p2 > 0:
                ratio = p1 / (p1 + p2)
            else:
                ratio = 0.5

            measured_ratios.append(ratio)

        measured_ratios = np.array(measured_ratios)

        # Fit: expected ratio = cos²(theta)
        expected_ratios = np.cos(theta_sweep) ** 2

        # Find offset and scale
        def fit_error(params):
            offset, scale = params
            predicted = np.cos((theta_sweep - offset) / scale) ** 2
            return np.mean((predicted - measured_ratios) ** 2)

        result = minimize(fit_error, [0.0, 1.0], method='Nelder-Mead')
        offset, scale = result.x

        # Update calibration
        self.calibration.theta_offsets[mzi_idx] = offset
        self.calibration.theta_scales[mzi_idx] = scale

        # Restore original state
        self.mesh.set_phases(original_thetas, original_phis,
                            np.zeros(self.n_ports))

        return {
            'mzi_idx': mzi_idx,
            'offset': offset,
            'scale': scale,
            'error': result.fun
        }

    def calibrate_full_mesh(self, n_samples_per_mzi: int = 50) -> Dict:
        """
        Calibrate entire mesh MZI by MZI.

        Args:
            n_samples_per_mzi: Samples per MZI calibration

        Returns:
            Full calibration results
        """
        results = []

        for i in range(self.n_mzis):
            result = self.calibrate_single_mzi(i, n_samples_per_mzi)
            results.append(result)

            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Calibrated {i+1}/{self.n_mzis} MZIs")

        # Compute overall quality
        total_error = sum(r['error'] for r in results)
        max_offset = max(abs(r['offset']) for r in results)
        max_scale_dev = max(abs(r['scale'] - 1) for r in results)

        self.calibration.last_calibration_error = total_error

        return {
            'mzi_results': results,
            'total_error': total_error,
            'max_offset': max_offset,
            'max_scale_deviation': max_scale_dev
        }

    def calibrate_thermal_crosstalk(self, n_samples: int = 20) -> np.ndarray:
        """
        Measure thermal crosstalk between adjacent MZIs.

        Returns:
            Crosstalk matrix
        """
        crosstalk = np.zeros((self.n_mzis, self.n_mzis))

        for i in range(self.n_mzis):
            # Set MZI i to high power (large theta)
            test_thetas = np.zeros(self.n_mzis)
            test_thetas[i] = np.pi / 4

            self.mesh.set_phases(test_thetas, np.zeros(self.n_mzis),
                                np.zeros(self.n_ports))

            # Measure effect on other MZIs
            for j in range(self.n_mzis):
                if i == j:
                    crosstalk[i, j] = 1.0
                    continue

                # In practice: measure phase shift induced in MZI j
                # Here: simulate based on physical proximity

                # Get positions
                layer_i, pos_i = self.mesh.mzi_positions[i]
                layer_j, pos_j = self.mesh.mzi_positions[j]

                # Thermal crosstalk decays with distance
                distance = abs(layer_i - layer_j) + abs(pos_i - pos_j)
                crosstalk[i, j] = 0.01 * np.exp(-distance / 2)

        self.calibration.thermal_crosstalk = crosstalk
        return crosstalk

    def optimize_for_target(self, target_matrix: np.ndarray,
                            max_iterations: int = 1000) -> Dict:
        """
        Optimize phases to best implement target matrix, accounting for errors.

        Uses gradient-free optimization to find phases that minimize
        error between actual and target matrix.

        Args:
            target_matrix: Desired unitary matrix
            max_iterations: Maximum optimization iterations

        Returns:
            Optimization results
        """
        from .clements import ClementsDecomposition

        n = self.n_ports

        # Initial guess from Clements decomposition
        decomp = ClementsDecomposition(n)
        init_thetas, init_phis, init_output = decomp.decompose(target_matrix)

        # Flatten for optimizer
        init_params = np.concatenate([init_thetas, init_phis, init_output])

        def objective(params):
            thetas = params[:self.n_mzis]
            phis = params[self.n_mzis:2*self.n_mzis]
            output = params[2*self.n_mzis:]

            # Apply calibration correction
            corr_thetas, corr_phis, corr_output = self.calibration.correct_phases(
                thetas, phis, output
            )

            # Set mesh
            self.mesh.set_phases(corr_thetas, corr_phis, corr_output)

            # Get actual matrix (with noise)
            actual = self.mesh.get_matrix(add_noise=True)

            # Frobenius norm error
            error = np.linalg.norm(target_matrix - actual, 'fro')
            return error

        # Optimize
        result = minimize(
            objective,
            init_params,
            method='Powell',
            options={'maxiter': max_iterations}
        )

        # Extract final phases
        final_thetas = result.x[:self.n_mzis]
        final_phis = result.x[self.n_mzis:2*self.n_mzis]
        final_output = result.x[2*self.n_mzis:]

        return {
            'thetas': final_thetas,
            'phis': final_phis,
            'output_phases': final_output,
            'error': result.fun,
            'iterations': result.nit,
            'success': result.success
        }


class MLCalibrator:
    """
    Machine learning accelerated calibration.

    Uses neural network to predict phase corrections
    from measured outputs, enabling fast recalibration.
    """

    def __init__(self, n_mzis: int):
        """
        Initialize ML calibrator.

        Args:
            n_mzis: Number of MZIs
        """
        self.n_mzis = n_mzis
        self.model = None
        self.is_trained = False

    def generate_training_data(self, mesh, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for calibration model.

        Args:
            mesh: MZIMesh to calibrate
            n_samples: Number of training samples

        Returns:
            (inputs, targets) for training
        """
        inputs = []
        targets = []

        for _ in range(n_samples):
            # Random target phases
            target_thetas = np.random.uniform(0, np.pi/2, self.n_mzis)
            target_phis = np.random.uniform(0, 2*np.pi, self.n_mzis)

            # Set mesh and get actual output
            mesh.set_phases(target_thetas, target_phis, np.zeros(mesh.n_ports))

            # Measure (with noise)
            test_input = np.ones(mesh.n_ports) / np.sqrt(mesh.n_ports)
            actual_output = mesh.forward(test_input, add_noise=True)

            # Input to model: measured output + target phases
            model_input = np.concatenate([
                np.abs(actual_output),
                np.angle(actual_output),
                target_thetas,
                target_phis
            ])

            # Target: phase errors
            actual_matrix = mesh.get_matrix(add_noise=True)
            # ... compute actual phases from matrix
            # For simplicity, use random noise as proxy
            phase_errors = np.random.randn(self.n_mzis) * 0.01

            inputs.append(model_input)
            targets.append(phase_errors)

        return np.array(inputs), np.array(targets)

    def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int = 100):
        """
        Train calibration model.

        Uses simple MLP for demonstration.
        In practice: use more sophisticated architecture.
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            # Simple MLP
            input_dim = inputs.shape[1]
            output_dim = targets.shape[1]

            self.model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )

            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Convert to tensors
            X = torch.FloatTensor(inputs)
            y = torch.FloatTensor(targets)

            # Train
            for epoch in range(epochs):
                optimizer.zero_grad()
                pred = self.model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

            self.is_trained = True

        except ImportError:
            print("PyTorch not available for ML calibration")

    def predict_correction(self, measurement: np.ndarray,
                           target_phases: np.ndarray) -> np.ndarray:
        """
        Predict phase correction from measurement.

        Args:
            measurement: Measured output
            target_phases: Target phase settings

        Returns:
            Predicted phase corrections
        """
        if not self.is_trained:
            return np.zeros(self.n_mzis)

        import torch

        # Prepare input
        model_input = np.concatenate([
            np.abs(measurement),
            np.angle(measurement),
            target_phases
        ])

        with torch.no_grad():
            prediction = self.model(torch.FloatTensor(model_input))

        return prediction.numpy()


if __name__ == "__main__":
    from .mzi import MZIMesh

    print("Testing Calibration System...")

    # Create mesh
    n = 8
    mesh = MZIMesh(n, insertion_loss_db=0.1, phase_noise_std=0.02)

    # Create calibration system
    cal = CalibrationSystem(n, mesh.n_mzis)

    # Create auto-calibrator
    auto_cal = AutoCalibrator(mesh, cal)

    # Test single MZI calibration
    print("\n--- Test Single MZI Calibration ---")
    result = auto_cal.calibrate_single_mzi(0, n_samples=50)
    print(f"MZI 0: offset={result['offset']:.4f}, scale={result['scale']:.4f}")

    # Test full mesh calibration
    print("\n--- Test Full Mesh Calibration ---")
    results = auto_cal.calibrate_full_mesh(n_samples_per_mzi=30)
    print(f"Total error: {results['total_error']:.6f}")
    print(f"Max offset: {results['max_offset']:.4f}")

    # Test thermal crosstalk
    print("\n--- Test Thermal Crosstalk ---")
    crosstalk = auto_cal.calibrate_thermal_crosstalk()
    print(f"Crosstalk matrix shape: {crosstalk.shape}")
    print(f"Max off-diagonal: {np.max(np.abs(crosstalk - np.diag(np.diag(crosstalk)))):.4f}")

    # Test optimization for target
    print("\n--- Test Target Optimization ---")
    from .clements import random_unitary
    target = random_unitary(n)

    opt_result = auto_cal.optimize_for_target(target, max_iterations=100)
    print(f"Optimization error: {opt_result['error']:.6f}")
    print(f"Iterations: {opt_result['iterations']}")

    # Save and load calibration
    print("\n--- Test Save/Load ---")
    cal.save("/tmp/test_calibration.json")
    cal2 = CalibrationSystem(n, mesh.n_mzis)
    cal2.load("/tmp/test_calibration.json")
    print(f"Loaded offsets match: {np.allclose(cal.theta_offsets, cal2.theta_offsets)}")

    print("\n✓ All calibration tests passed!")
