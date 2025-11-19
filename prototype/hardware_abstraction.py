"""
PhotonCore-X Hardware Abstraction Layer (HAL)

Unified interface for all prototype targets:
- Software simulation (phase1_physics)
- FPGA emulation (phase2_fpga)
- Bench-top optics (phase3_benchtop)
- Foundry chip (phase4_foundry)

Usage:
    from hardware_abstraction import PhotonCoreHAL, DeviceType

    # Software simulation
    hal = PhotonCoreHAL(DeviceType.SIMULATION, n_ports=64)

    # FPGA emulation
    hal = PhotonCoreHAL(DeviceType.FPGA, n_ports=64, fpga_ip="192.168.1.100")

    # Bench-top
    hal = PhotonCoreHAL(DeviceType.BENCHTOP, n_ports=8, serial_port="/dev/ttyUSB0")

    # Real chip
    hal = PhotonCoreHAL(DeviceType.ASIC, n_ports=64, device_path="/dev/photoncore0")

    # Use uniformly
    hal.load_matrix(weights)
    output = hal.forward(input_vector)
"""

import numpy as np
from enum import Enum
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
import time
import sys
import os

# Add prototype paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase1_physics'))
sys.path.insert(0, os.path.dirname(__file__))


class DeviceType(Enum):
    """Target device types."""
    SIMULATION = "simulation"    # Software physics simulation
    FPGA = "fpga"                # FPGA emulation
    BENCHTOP = "benchtop"        # Tabletop optics
    ASIC = "asic"                # Foundry chip


class PhotonCoreBackend(ABC):
    """Abstract base class for all backends."""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the device."""
        pass

    @abstractmethod
    def load_matrix(self, matrix: np.ndarray) -> None:
        """Load weight matrix for optical MVM."""
        pass

    @abstractmethod
    def forward(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Perform forward pass."""
        pass

    @abstractmethod
    def calibrate(self) -> Dict[str, Any]:
        """Run calibration routine."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get device status."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown device."""
        pass


# =============================================================================
# Simulation Backend
# =============================================================================

class SimulationBackend(PhotonCoreBackend):
    """Software physics simulation backend."""

    def __init__(self, n_ports: int, **kwargs):
        self.n_ports = n_ports
        self.mesh = None
        self.matrix_loaded = False

        # Import realistic physics
        from realistic_mzi import RealisticMZIMesh, FabricationParams, ThermalParams

        # Use realistic parameters
        fab = FabricationParams(
            width_variation_nm=kwargs.get('width_var', 5.0),
            thickness_variation_nm=kwargs.get('thickness_var', 2.0)
        )
        thermal = ThermalParams(
            thermal_tau_us=kwargs.get('thermal_tau', 10.0)
        )

        self.mesh = RealisticMZIMesh(n_ports, fab, thermal)

    def initialize(self) -> bool:
        return True

    def load_matrix(self, matrix: np.ndarray) -> None:
        """Decompose and load matrix."""
        # Add parent path for clements
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'photoncore_prototype', 'src'))
        from clements import ClementsDecomposition

        assert matrix.shape == (self.n_ports, self.n_ports)

        # Decompose
        decomp = ClementsDecomposition(self.n_ports)
        thetas, phis, output_phases = decomp.decompose(matrix)

        # Load into mesh
        self.mesh.set_phases(thetas, phis, output_phases)
        self.matrix_loaded = True

    def forward(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        if not self.matrix_loaded:
            raise RuntimeError("No matrix loaded")

        return self.mesh.forward(x.astype(np.complex128), add_noise=add_noise)

    def calibrate(self) -> Dict[str, Any]:
        """Run calibration (in simulation, just measure fab variations)."""
        results = {
            'success': True,
            'n_mzis': self.mesh.n_mzis,
            'power_mw': self.mesh.get_total_power_mw(),
            'fab_variations': []
        }

        for i, mzi in enumerate(self.mesh.mzis):
            results['fab_variations'].append({
                'mzi_idx': i,
                'dn_eff': mzi.dn_eff_fab,
                'split_error': mzi.splitting_error
            })

        return results

    def get_status(self) -> Dict[str, Any]:
        return {
            'device_type': 'simulation',
            'n_ports': self.n_ports,
            'n_mzis': self.mesh.n_mzis,
            'matrix_loaded': self.matrix_loaded,
            'power_mw': self.mesh.get_total_power_mw() if self.matrix_loaded else 0
        }

    def shutdown(self) -> None:
        pass


# =============================================================================
# FPGA Backend
# =============================================================================

class FPGABackend(PhotonCoreBackend):
    """FPGA emulation backend (TCP/IP communication)."""

    def __init__(self, n_ports: int, **kwargs):
        self.n_ports = n_ports
        self.fpga_ip = kwargs.get('fpga_ip', '192.168.1.100')
        self.fpga_port = kwargs.get('fpga_port', 5000)
        self.socket = None
        self.connected = False

    def initialize(self) -> bool:
        """Connect to FPGA over TCP."""
        import socket

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.fpga_ip, self.fpga_port))
            self.connected = True

            # Send init command
            self._send_command(0x00, bytes([self.n_ports]))
            response = self._recv_response()

            return response[0] == 0x01  # ACK

        except Exception as e:
            print(f"FPGA connection failed: {e}")
            return False

    def _send_command(self, cmd: int, data: bytes):
        """Send command to FPGA."""
        packet = bytes([cmd, len(data)]) + data
        self.socket.send(packet)

    def _recv_response(self) -> bytes:
        """Receive response from FPGA."""
        return self.socket.recv(1024)

    def load_matrix(self, matrix: np.ndarray) -> None:
        """Send matrix to FPGA."""
        # Decompose to phase settings
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'photoncore_prototype', 'src'))
        from clements import ClementsDecomposition

        decomp = ClementsDecomposition(self.n_ports)
        thetas, phis, output_phases = decomp.decompose(matrix)

        # Convert to 16-bit DAC values
        dac_values = []
        for theta, phi in zip(thetas, phis):
            theta_dac = int(theta / np.pi * 32767)
            phi_dac = int(phi / (2 * np.pi) * 65535)
            dac_values.extend([theta_dac >> 8, theta_dac & 0xFF])
            dac_values.extend([phi_dac >> 8, phi_dac & 0xFF])

        # Send to FPGA
        self._send_command(0x01, bytes(dac_values))
        self._recv_response()

    def forward(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Run forward pass on FPGA."""
        # Convert input to DAC values
        input_dac = []
        for val in x:
            amp = int(np.abs(val) * 65535)
            phase = int(np.angle(val) / (2 * np.pi) * 65535)
            input_dac.extend([amp >> 8, amp & 0xFF])
            input_dac.extend([phase >> 8, phase & 0xFF])

        # Send and trigger
        self._send_command(0x02, bytes(input_dac))

        # Wait for completion
        time.sleep(0.001)  # 1ms for thermal settling

        # Read ADC values
        self._send_command(0x03, bytes([]))
        response = self._recv_response()

        # Convert ADC to complex
        output = np.zeros(self.n_ports, dtype=np.complex128)
        for i in range(self.n_ports):
            amp = (response[i*4] << 8 | response[i*4 + 1]) / 65535
            phase = (response[i*4 + 2] << 8 | response[i*4 + 3]) / 65535 * 2 * np.pi
            output[i] = amp * np.exp(1j * phase)

        return output

    def calibrate(self) -> Dict[str, Any]:
        """Run FPGA calibration sequence."""
        self._send_command(0x05, bytes([]))

        # Wait for calibration (can take minutes)
        time.sleep(10)

        response = self._recv_response()
        return {'success': response[0] == 0x01}

    def get_status(self) -> Dict[str, Any]:
        self._send_command(0x06, bytes([]))
        response = self._recv_response()

        return {
            'device_type': 'fpga',
            'connected': self.connected,
            'n_ports': self.n_ports,
            'temperature': response[0],
            'power_mw': (response[1] << 8 | response[2])
        }

    def shutdown(self) -> None:
        if self.socket:
            self.socket.close()


# =============================================================================
# Bench-top Backend
# =============================================================================

class BenchtopBackend(PhotonCoreBackend):
    """Bench-top optics control via serial/GPIB."""

    def __init__(self, n_ports: int, **kwargs):
        self.n_ports = n_ports
        self.serial_port = kwargs.get('serial_port', '/dev/ttyUSB0')
        self.serial = None

    def initialize(self) -> bool:
        """Connect to control electronics."""
        try:
            import serial
            self.serial = serial.Serial(
                self.serial_port,
                baudrate=115200,
                timeout=1.0
            )
            return True
        except Exception as e:
            print(f"Benchtop connection failed: {e}")
            return False

    def load_matrix(self, matrix: np.ndarray) -> None:
        """Set phase shifter voltages."""
        # Similar to FPGA but over serial
        pass

    def forward(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Measure optical output."""
        # Set input modulators
        # Trigger measurement
        # Read photodetector outputs
        pass

    def calibrate(self) -> Dict[str, Any]:
        """Calibrate bench-top system."""
        # Sweep each phase shifter
        # Record power meter readings
        # Fit response curves
        pass

    def get_status(self) -> Dict[str, Any]:
        return {
            'device_type': 'benchtop',
            'n_ports': self.n_ports,
            'serial_port': self.serial_port
        }

    def shutdown(self) -> None:
        if self.serial:
            self.serial.close()


# =============================================================================
# ASIC Backend
# =============================================================================

class ASICBackend(PhotonCoreBackend):
    """Real PhotonCore chip via driver."""

    def __init__(self, n_ports: int, **kwargs):
        self.n_ports = n_ports
        self.device_path = kwargs.get('device_path', '/dev/photoncore0')
        self.device = None

    def initialize(self) -> bool:
        """Open device driver."""
        try:
            self.device = open(self.device_path, 'r+b', buffering=0)
            return True
        except Exception as e:
            print(f"ASIC device open failed: {e}")
            return False

    def load_matrix(self, matrix: np.ndarray) -> None:
        """Write matrix to chip via DMA."""
        # ioctl to configure DMA
        # Write decomposed phases
        pass

    def forward(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Hardware forward pass."""
        # Write input
        # Trigger
        # Read output via DMA
        pass

    def calibrate(self) -> Dict[str, Any]:
        """Run on-chip calibration."""
        pass

    def get_status(self) -> Dict[str, Any]:
        return {
            'device_type': 'asic',
            'n_ports': self.n_ports,
            'device_path': self.device_path
        }

    def shutdown(self) -> None:
        if self.device:
            self.device.close()


# =============================================================================
# Unified HAL
# =============================================================================

class PhotonCoreHAL:
    """
    Unified Hardware Abstraction Layer for PhotonCore-X.

    Provides consistent interface across all device types.
    """

    def __init__(self, device_type: DeviceType, n_ports: int = 64, **kwargs):
        self.device_type = device_type
        self.n_ports = n_ports

        # Create backend
        if device_type == DeviceType.SIMULATION:
            self.backend = SimulationBackend(n_ports, **kwargs)
        elif device_type == DeviceType.FPGA:
            self.backend = FPGABackend(n_ports, **kwargs)
        elif device_type == DeviceType.BENCHTOP:
            self.backend = BenchtopBackend(n_ports, **kwargs)
        elif device_type == DeviceType.ASIC:
            self.backend = ASICBackend(n_ports, **kwargs)
        else:
            raise ValueError(f"Unknown device type: {device_type}")

        # Initialize
        if not self.backend.initialize():
            raise RuntimeError(f"Failed to initialize {device_type.value}")

    def load_matrix(self, matrix: np.ndarray) -> None:
        """Load weight matrix."""
        self.backend.load_matrix(matrix)

    def forward(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Forward pass through optical mesh."""
        return self.backend.forward(x, add_noise)

    def calibrate(self) -> Dict[str, Any]:
        """Run calibration."""
        return self.backend.calibrate()

    def status(self) -> Dict[str, Any]:
        """Get device status."""
        return self.backend.get_status()

    def close(self):
        """Shutdown device."""
        self.backend.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# Test
# =============================================================================

def test_hal():
    """Test HAL with simulation backend."""
    print("PhotonCore-X HAL Test")
    print("=" * 50)

    # Create HAL with simulation
    with PhotonCoreHAL(DeviceType.SIMULATION, n_ports=8) as hal:

        # Create random unitary matrix
        from scipy.stats import unitary_group
        U = unitary_group.rvs(8)

        # Load matrix
        hal.load_matrix(U)
        print(f"\nLoaded 8x8 unitary matrix")

        # Forward pass
        x = np.zeros(8, dtype=np.complex128)
        x[0] = 1.0

        y = hal.forward(x, add_noise=False)
        print(f"\nInput: {x}")
        print(f"Output norm: {np.linalg.norm(y):.4f}")

        # Status
        status = hal.status()
        print(f"\nStatus: {status}")

        # Calibration
        cal = hal.calibrate()
        print(f"\nCalibration: {cal['n_mzis']} MZIs, {cal['power_mw']:.1f} mW")

    print("\n" + "=" * 50)
    print("HAL test complete!")


if __name__ == "__main__":
    test_hal()
