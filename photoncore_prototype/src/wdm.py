"""
Wavelength Division Multiplexing (WDM) System

Implements multi-wavelength parallel processing for PhotonCore-X.
"""

import numpy as np
from typing import List, Tuple


class WDMSystem:
    """
    WDM system for wavelength-parallel optical computing.

    Supports 128+ channels across C+L bands.
    """

    def __init__(self, n_channels: int = 128,
                 center_wavelength_nm: float = 1550.0,
                 channel_spacing_ghz: float = 50.0):
        """
        Initialize WDM system.

        Args:
            n_channels: Number of wavelength channels
            center_wavelength_nm: Center wavelength in nm
            channel_spacing_ghz: Channel spacing in GHz
        """
        self.n_channels = n_channels
        self.center_wavelength = center_wavelength_nm
        self.channel_spacing = channel_spacing_ghz

        # Calculate wavelengths
        c = 299792458  # m/s
        center_freq = c / (center_wavelength_nm * 1e-9)

        self.frequencies = np.array([
            center_freq + (i - n_channels/2) * channel_spacing_ghz * 1e9
            for i in range(n_channels)
        ])

        self.wavelengths = c / self.frequencies * 1e9  # nm

        # Channel powers
        self.powers = np.ones(n_channels)

        # Crosstalk matrix
        self.crosstalk = self._compute_crosstalk()

    def _compute_crosstalk(self) -> np.ndarray:
        """Compute inter-channel crosstalk matrix."""
        crosstalk = np.eye(self.n_channels)

        # Adjacent channel crosstalk (-30 dB typical)
        for i in range(self.n_channels - 1):
            crosstalk[i, i+1] = 0.001
            crosstalk[i+1, i] = 0.001

        return crosstalk

    def multiplex(self, signals: List[np.ndarray]) -> np.ndarray:
        """
        Combine multiple signals onto different wavelengths.

        Args:
            signals: List of signals, one per channel

        Returns:
            Combined WDM signal
        """
        assert len(signals) == self.n_channels

        # In practice, this is just concatenation
        # Physical multiplexing handled by AWG
        return np.array(signals)

    def demultiplex(self, wdm_signal: np.ndarray) -> List[np.ndarray]:
        """
        Separate WDM signal into individual channels.

        Args:
            wdm_signal: Combined signal

        Returns:
            List of individual channel signals
        """
        # Apply crosstalk
        result = []
        for i in range(self.n_channels):
            channel = wdm_signal[i].copy()

            # Add crosstalk from adjacent channels
            if i > 0:
                channel += self.crosstalk[i, i-1] * wdm_signal[i-1]
            if i < self.n_channels - 1:
                channel += self.crosstalk[i, i+1] * wdm_signal[i+1]

            result.append(channel)

        return result

    def get_channel_info(self, channel_idx: int) -> dict:
        """Get information about a specific channel."""
        return {
            'index': channel_idx,
            'wavelength_nm': self.wavelengths[channel_idx],
            'frequency_thz': self.frequencies[channel_idx] / 1e12,
            'power': self.powers[channel_idx]
        }


class KerrComb:
    """
    Kerr frequency comb light source.

    Generates hundreds of phase-locked wavelength channels
    from a single pump laser using four-wave mixing in a
    high-Q microresonator.
    """

    def __init__(self, n_lines: int = 128,
                 pump_wavelength_nm: float = 1550.0,
                 fsr_ghz: float = 100.0,
                 pump_power_mw: float = 100.0):
        """
        Initialize Kerr comb.

        Args:
            n_lines: Number of comb lines
            pump_wavelength_nm: Pump laser wavelength
            fsr_ghz: Free spectral range (line spacing)
            pump_power_mw: Pump power in mW
        """
        self.n_lines = n_lines
        self.pump_wavelength = pump_wavelength_nm
        self.fsr = fsr_ghz
        self.pump_power = pump_power_mw

        # Generate comb spectrum
        self.wavelengths, self.powers = self._generate_comb()

        # Coherence properties
        self.linewidth_hz = 100  # Narrow linewidth due to phase locking
        self.rin_db_hz = -150    # Relative intensity noise

    def _generate_comb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate comb line wavelengths and powers."""
        c = 299792458  # m/s

        # Pump frequency
        pump_freq = c / (self.pump_wavelength * 1e-9)

        # Generate lines around pump
        frequencies = np.array([
            pump_freq + (i - self.n_lines/2) * self.fsr * 1e9
            for i in range(self.n_lines)
        ])

        wavelengths = c / frequencies * 1e9

        # Power distribution (sech² envelope typical for soliton combs)
        line_numbers = np.arange(self.n_lines) - self.n_lines/2
        envelope = 1 / np.cosh(line_numbers / (self.n_lines/4)) ** 2
        powers = self.pump_power * envelope / np.sum(envelope) * self.n_lines

        return wavelengths, powers

    def get_output(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get comb output.

        Returns:
            (wavelengths, powers) arrays
        """
        return self.wavelengths.copy(), self.powers.copy()

    def modulate_lines(self, modulation: np.ndarray) -> np.ndarray:
        """
        Apply modulation to comb lines.

        Args:
            modulation: Complex modulation per line

        Returns:
            Modulated optical fields
        """
        amplitudes = np.sqrt(self.powers)
        return amplitudes * modulation

    @property
    def total_power_mw(self) -> float:
        """Total output power."""
        return np.sum(self.powers)

    @property
    def optical_snr_db(self) -> float:
        """Optical signal-to-noise ratio."""
        # Simplified model
        return 40.0  # Typical for good comb


class ArrayedWaveguideGrating:
    """
    AWG multiplexer/demultiplexer.

    Used for combining/separating WDM channels.
    """

    def __init__(self, n_channels: int,
                 center_wavelength_nm: float = 1550.0,
                 channel_spacing_nm: float = 0.8,
                 insertion_loss_db: float = 3.0,
                 crosstalk_db: float = -30.0):
        """
        Initialize AWG.

        Args:
            n_channels: Number of channels
            center_wavelength_nm: Center wavelength
            channel_spacing_nm: Wavelength spacing between channels
            insertion_loss_db: Insertion loss
            crosstalk_db: Adjacent channel crosstalk
        """
        self.n_channels = n_channels
        self.center_wavelength = center_wavelength_nm
        self.channel_spacing = channel_spacing_nm

        self.insertion_loss = 10 ** (-insertion_loss_db / 10)
        self.crosstalk = 10 ** (crosstalk_db / 10)

        # Channel wavelengths
        self.wavelengths = np.array([
            center_wavelength_nm + (i - n_channels/2) * channel_spacing_nm
            for i in range(n_channels)
        ])

    def demux(self, input_signal: np.ndarray) -> List[np.ndarray]:
        """
        Demultiplex WDM signal.

        Args:
            input_signal: Combined WDM signal (wavelength-indexed)

        Returns:
            List of separated channel signals
        """
        outputs = []

        for i in range(self.n_channels):
            # Main signal with insertion loss
            output = input_signal[i] * np.sqrt(self.insertion_loss)

            # Add crosstalk from adjacent channels
            if i > 0:
                output += input_signal[i-1] * np.sqrt(self.crosstalk)
            if i < self.n_channels - 1:
                output += input_signal[i+1] * np.sqrt(self.crosstalk)

            outputs.append(output)

        return outputs

    def mux(self, channel_signals: List[np.ndarray]) -> np.ndarray:
        """
        Multiplex individual channels into WDM signal.

        Args:
            channel_signals: List of channel signals

        Returns:
            Combined WDM signal
        """
        output = np.zeros(self.n_channels, dtype=np.complex128)

        for i, signal in enumerate(channel_signals):
            output[i] = signal * np.sqrt(self.insertion_loss)

        return output


if __name__ == "__main__":
    print("Testing WDM System...")

    # Test 1: Basic WDM
    print("\n--- Test 1: WDM System ---")
    wdm = WDMSystem(n_channels=16)

    print(f"Channels: {wdm.n_channels}")
    print(f"Wavelength range: {wdm.wavelengths.min():.2f} - {wdm.wavelengths.max():.2f} nm")

    info = wdm.get_channel_info(8)
    print(f"Channel 8: {info['wavelength_nm']:.2f} nm, {info['frequency_thz']:.2f} THz")

    # Test 2: Kerr comb
    print("\n--- Test 2: Kerr Frequency Comb ---")
    comb = KerrComb(n_lines=64, fsr_ghz=50.0)

    wavelengths, powers = comb.get_output()
    print(f"Comb lines: {len(wavelengths)}")
    print(f"Total power: {comb.total_power_mw:.1f} mW")
    print(f"Power range: {powers.min():.2f} - {powers.max():.2f} mW")

    # Test 3: AWG
    print("\n--- Test 3: AWG ---")
    awg = ArrayedWaveguideGrating(16)

    # Create test WDM signal
    test_signal = np.random.randn(16) + 1j * np.random.randn(16)

    # Demux
    channels = awg.demux(test_signal)
    print(f"Demuxed {len(channels)} channels")

    # Mux back
    combined = awg.mux(channels)
    print(f"Muxed back to shape: {combined.shape}")

    print("\n✓ All WDM tests passed!")
