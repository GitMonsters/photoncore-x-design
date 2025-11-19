"""
Phase 1: Enhanced Physics Simulator

Realistic MZI mesh with:
- Temperature-dependent phase drift
- Fabrication tolerances
- Wavelength dispersion
- Crosstalk coupling
- Shot noise and thermal noise
- Insertion loss accumulation
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class FabricationParams:
    """Silicon photonics fabrication tolerances (IMEC/GlobalFoundries)."""

    # Waveguide width variation (nm, 3-sigma)
    width_variation_nm: float = 5.0

    # Thickness variation (nm, 3-sigma)
    thickness_variation_nm: float = 2.0

    # Effective index variation from width
    dn_eff_per_nm_width: float = 0.001

    # Effective index variation from thickness
    dn_eff_per_nm_thickness: float = 0.002

    # Coupler gap variation (nm, 3-sigma)
    gap_variation_nm: float = 3.0

    # Heater resistance variation (%, 3-sigma)
    heater_resistance_var: float = 5.0


@dataclass
class ThermalParams:
    """Thermal properties for phase shifters."""

    # Thermo-optic coefficient (1/K) for silicon
    dn_dT: float = 1.86e-4

    # Heater efficiency (K/mW)
    heater_efficiency: float = 25.0

    # Thermal time constant (μs)
    thermal_tau_us: float = 10.0

    # Ambient temperature variation (K, 3-sigma)
    ambient_variation_K: float = 0.5

    # Cross-heater thermal coupling
    thermal_coupling: float = 0.05


@dataclass
class OpticalParams:
    """Optical properties."""

    # Center wavelength (nm)
    wavelength_nm: float = 1550.0

    # Group index for silicon waveguide
    n_group: float = 4.2

    # Propagation loss (dB/cm)
    propagation_loss_db_cm: float = 2.0

    # Bend loss per 90° bend (dB)
    bend_loss_db: float = 0.01

    # Coupler excess loss (dB)
    coupler_loss_db: float = 0.1

    # Wavelength dispersion of n_eff (1/nm)
    dn_eff_dlambda: float = -1e-4


class RealisticMZI:
    """
    Single MZI with realistic physics.

    Includes fabrication variations, thermal effects, and noise.
    """

    def __init__(self,
                 arm_length_um: float = 100.0,
                 fab_params: Optional[FabricationParams] = None,
                 thermal_params: Optional[ThermalParams] = None,
                 optical_params: Optional[OpticalParams] = None):

        self.arm_length_um = arm_length_um
        self.fab = fab_params or FabricationParams()
        self.thermal = thermal_params or ThermalParams()
        self.optical = optical_params or OpticalParams()

        # Generate fabrication variations (fixed at creation)
        self._generate_fab_variations()

        # State
        self.heater_power_mw = 0.0
        self.temperature_K = 300.0
        self._target_phase = 0.0
        self._external_phase = 0.0

    def _generate_fab_variations(self):
        """Generate random fabrication variations for this MZI."""
        # Width variation
        dw = np.random.normal(0, self.fab.width_variation_nm / 3)
        self.dn_eff_width = dw * self.fab.dn_eff_per_nm_width

        # Thickness variation
        dt = np.random.normal(0, self.fab.thickness_variation_nm / 3)
        self.dn_eff_thickness = dt * self.fab.dn_eff_per_nm_thickness

        # Total effective index variation
        self.dn_eff_fab = self.dn_eff_width + self.dn_eff_thickness

        # Coupler splitting ratio variation
        dg = np.random.normal(0, self.fab.gap_variation_nm / 3)
        # Splitting ratio deviation from 50:50
        self.splitting_error = 0.01 * dg  # ~1% per nm

        # Heater resistance variation
        self.heater_resistance_factor = 1.0 + np.random.normal(
            0, self.fab.heater_resistance_var / 300
        )

    def set_phase(self, theta: float, phi: float):
        """Set target phases for MZI."""
        self._target_phase = theta
        self._external_phase = phi

        # Calculate required heater power
        # Phase = 2π * n_eff * L / λ
        # Δphase = 2π * (dn/dT) * ΔT * L / λ
        wavelength_m = self.optical.wavelength_nm * 1e-9
        length_m = self.arm_length_um * 1e-6

        # Required temperature change
        delta_T = theta * wavelength_m / (2 * np.pi * self.thermal.dn_dT * length_m)

        # Required heater power (accounting for resistance variation)
        self.heater_power_mw = delta_T / self.thermal.heater_efficiency
        self.heater_power_mw *= self.heater_resistance_factor

    def get_actual_phase(self, wavelength_nm: Optional[float] = None) -> Tuple[float, float]:
        """
        Get actual phase including all error sources.

        Returns (theta_actual, phi_actual).
        """
        if wavelength_nm is None:
            wavelength_nm = self.optical.wavelength_nm

        wavelength_m = wavelength_nm * 1e-9
        length_m = self.arm_length_um * 1e-6

        # Base phase from heater
        actual_T = self.heater_power_mw * self.thermal.heater_efficiency
        actual_T /= self.heater_resistance_factor  # Undo resistance factor effect

        theta_thermal = 2 * np.pi * self.thermal.dn_dT * actual_T * length_m / wavelength_m

        # Fabrication-induced phase error
        theta_fab = 2 * np.pi * self.dn_eff_fab * length_m / wavelength_m

        # Wavelength dispersion
        dlambda = wavelength_nm - self.optical.wavelength_nm
        theta_dispersion = 2 * np.pi * self.optical.dn_eff_dlambda * dlambda * length_m / wavelength_m

        # Ambient temperature variation
        ambient_var = np.random.normal(0, self.thermal.ambient_variation_K / 3)
        theta_ambient = 2 * np.pi * self.thermal.dn_dT * ambient_var * length_m / wavelength_m

        theta_actual = theta_thermal + theta_fab + theta_dispersion + theta_ambient
        phi_actual = self._external_phase  # External phase shifter (assume ideal for now)

        return theta_actual, phi_actual

    def get_transfer_matrix(self, wavelength_nm: Optional[float] = None) -> np.ndarray:
        """
        Get 2x2 transfer matrix with realistic effects.

        M = [[t, r], [r*, t*]] with splitting errors
        """
        theta, phi = self.get_actual_phase(wavelength_nm)

        # Ideal splitting
        cos_t = np.cos(theta / 2)
        sin_t = np.sin(theta / 2)

        # Add splitting ratio error
        split_err = self.splitting_error
        cos_t_eff = cos_t * (1 - split_err)
        sin_t_eff = sin_t * (1 + split_err)

        # Normalize to maintain unitarity (approximately)
        norm = np.sqrt(cos_t_eff**2 + sin_t_eff**2)
        cos_t_eff /= norm
        sin_t_eff /= norm

        # Phase
        exp_phi = np.exp(1j * phi)

        # Transfer matrix
        M = np.array([
            [exp_phi * cos_t_eff, 1j * exp_phi * sin_t_eff],
            [1j * sin_t_eff, cos_t_eff]
        ], dtype=np.complex128)

        return M

    def get_insertion_loss(self) -> float:
        """Get total insertion loss in dB."""
        # Propagation loss
        prop_loss = self.optical.propagation_loss_db_cm * (self.arm_length_um / 1e4)

        # Coupler loss (2 couplers per MZI)
        coupler_loss = 2 * self.optical.coupler_loss_db

        # Bend losses (assume 2 bends)
        bend_loss = 2 * self.optical.bend_loss_db

        return prop_loss + coupler_loss + bend_loss


class RealisticMZIMesh:
    """
    Complete MZI mesh with realistic physics simulation.
    """

    def __init__(self,
                 n_ports: int,
                 fab_params: Optional[FabricationParams] = None,
                 thermal_params: Optional[ThermalParams] = None,
                 optical_params: Optional[OpticalParams] = None):

        self.n_ports = n_ports
        self.n_mzis = n_ports * (n_ports - 1) // 2

        self.fab = fab_params or FabricationParams()
        self.thermal = thermal_params or ThermalParams()
        self.optical = optical_params or OpticalParams()

        # Create MZI array with individual variations
        self.mzis = []
        for _ in range(self.n_mzis):
            mzi = RealisticMZI(
                arm_length_um=100.0,
                fab_params=self.fab,
                thermal_params=self.thermal,
                optical_params=self.optical
            )
            self.mzis.append(mzi)

        # Output phase shifters
        self.output_phases = np.zeros(n_ports)

        # Thermal crosstalk matrix
        self._build_thermal_crosstalk()

    def _build_thermal_crosstalk(self):
        """Build thermal coupling matrix between MZIs."""
        self.thermal_matrix = np.eye(self.n_mzis)

        # Adjacent MZIs have thermal coupling
        for i in range(self.n_mzis - 1):
            coupling = self.thermal.thermal_coupling
            self.thermal_matrix[i, i+1] = coupling
            self.thermal_matrix[i+1, i] = coupling

    def set_phases(self, thetas: list, phis: list, output_phases: list):
        """Set all MZI phases."""
        assert len(thetas) == self.n_mzis
        assert len(phis) == self.n_mzis
        assert len(output_phases) == self.n_ports

        for i, mzi in enumerate(self.mzis):
            mzi.set_phase(thetas[i], phis[i])

        self.output_phases = np.array(output_phases)

    def forward(self,
                x: np.ndarray,
                wavelength_nm: Optional[float] = None,
                add_noise: bool = True) -> np.ndarray:
        """
        Forward pass through mesh with realistic physics.
        """
        assert len(x) == self.n_ports

        state = x.astype(np.complex128).copy()

        # Apply thermal crosstalk to heater powers
        heater_powers = np.array([mzi.heater_power_mw for mzi in self.mzis])
        effective_powers = self.thermal_matrix @ heater_powers

        # Temporarily update MZI powers for this forward pass
        original_powers = heater_powers.copy()
        for i, mzi in enumerate(self.mzis):
            mzi.heater_power_mw = effective_powers[i]

        # Clements mesh structure
        mzi_idx = 0
        total_loss_db = 0.0

        for col in range(self.n_ports - 1):
            if col % 2 == 0:
                # Even column: pairs (0,1), (2,3), ...
                for row in range(0, self.n_ports - 1, 2):
                    if mzi_idx < self.n_mzis:
                        M = self.mzis[mzi_idx].get_transfer_matrix(wavelength_nm)
                        total_loss_db += self.mzis[mzi_idx].get_insertion_loss()

                        # Apply to state
                        temp = np.array([state[row], state[row + 1]])
                        result = M @ temp
                        state[row] = result[0]
                        state[row + 1] = result[1]

                        mzi_idx += 1
            else:
                # Odd column: pairs (1,2), (3,4), ...
                for row in range(1, self.n_ports - 1, 2):
                    if mzi_idx < self.n_mzis:
                        M = self.mzis[mzi_idx].get_transfer_matrix(wavelength_nm)
                        total_loss_db += self.mzis[mzi_idx].get_insertion_loss()

                        temp = np.array([state[row], state[row + 1]])
                        result = M @ temp
                        state[row] = result[0]
                        state[row + 1] = result[1]

                        mzi_idx += 1

        # Restore original heater powers
        for i, mzi in enumerate(self.mzis):
            mzi.heater_power_mw = original_powers[i]

        # Output phases
        state = state * np.exp(1j * self.output_phases)

        # Apply insertion loss
        loss_linear = 10 ** (-total_loss_db / 20)
        state = state * loss_linear

        # Add noise
        if add_noise:
            state = self._add_detection_noise(state)

        return state

    def _add_detection_noise(self, state: np.ndarray) -> np.ndarray:
        """Add shot noise and thermal noise to detected signal."""
        intensity = np.abs(state) ** 2

        # Shot noise (Poisson statistics)
        # Assume 1e6 photons per unit intensity
        n_photons = intensity * 1e6
        shot_noise_var = n_photons  # Poisson variance
        shot_noise = np.random.normal(0, np.sqrt(shot_noise_var)) / 1e6

        # Thermal noise (Johnson-Nyquist)
        thermal_noise_std = 1e-4  # Relative to signal
        thermal_noise = np.random.normal(0, thermal_noise_std, self.n_ports)

        # Apply to amplitude
        noisy_intensity = intensity + shot_noise + thermal_noise
        noisy_intensity = np.maximum(noisy_intensity, 0)  # Clip negative

        # Reconstruct with original phase
        phase = np.angle(state)
        return np.sqrt(noisy_intensity) * np.exp(1j * phase)

    def get_total_power_mw(self) -> float:
        """Get total heater power consumption."""
        return sum(mzi.heater_power_mw for mzi in self.mzis)

    def calibrate(self, target_matrix: np.ndarray, n_iterations: int = 100) -> dict:
        """
        Calibrate mesh to implement target matrix.

        Uses gradient-free optimization to account for fab variations.
        Returns calibration results.
        """
        from scipy.optimize import minimize

        # Initial guess from ideal decomposition
        # (Would use Clements decomposition here)
        n_params = 2 * self.n_mzis + self.n_ports
        x0 = np.zeros(n_params)

        def objective(params):
            thetas = params[:self.n_mzis]
            phis = params[self.n_mzis:2*self.n_mzis]
            output = params[2*self.n_mzis:]

            self.set_phases(thetas.tolist(), phis.tolist(), output.tolist())

            # Compute actual matrix
            actual = np.zeros((self.n_ports, self.n_ports), dtype=np.complex128)
            for i in range(self.n_ports):
                x = np.zeros(self.n_ports, dtype=np.complex128)
                x[i] = 1.0
                actual[:, i] = self.forward(x, add_noise=False)

            # Frobenius norm error
            error = np.linalg.norm(actual - target_matrix)
            return error

        result = minimize(
            objective,
            x0,
            method='Powell',
            options={'maxiter': n_iterations}
        )

        return {
            'success': result.success,
            'error': result.fun,
            'iterations': result.nit,
            'params': result.x
        }


def test_realistic_mesh():
    """Test realistic MZI mesh."""
    print("Testing Realistic MZI Mesh")
    print("=" * 50)

    # Create mesh with default parameters
    mesh = RealisticMZIMesh(n_ports=8)

    # Set random phases
    thetas = [np.random.uniform(0, np.pi) for _ in range(mesh.n_mzis)]
    phis = [np.random.uniform(0, 2*np.pi) for _ in range(mesh.n_mzis)]
    output = [np.random.uniform(0, 2*np.pi) for _ in range(8)]

    mesh.set_phases(thetas, phis, output)

    # Test forward pass
    x = np.zeros(8, dtype=np.complex128)
    x[0] = 1.0

    y_ideal = mesh.forward(x, add_noise=False)
    y_noisy = mesh.forward(x, add_noise=True)

    print(f"\nInput: {x}")
    print(f"\nOutput (no noise): norm = {np.linalg.norm(y_ideal):.4f}")
    print(f"Output (with noise): norm = {np.linalg.norm(y_noisy):.4f}")

    # Check power consumption
    power = mesh.get_total_power_mw()
    print(f"\nTotal heater power: {power:.2f} mW")

    # Test wavelength sensitivity
    wavelengths = [1540, 1550, 1560]
    print("\nWavelength sensitivity:")
    for wl in wavelengths:
        y = mesh.forward(x, wavelength_nm=wl, add_noise=False)
        print(f"  {wl} nm: norm = {np.linalg.norm(y):.4f}")

    print("\n" + "=" * 50)
    print("Test complete!")


if __name__ == "__main__":
    test_realistic_mesh()
