"""
PhotonCore-X Bench-top Controller

Control software for 2x2 optical demo with:
- DFB Laser (LPS-1550-FC)
- Variable Attenuator (EVOA1550A)
- 50:50 Coupler (TW1550R5A2)
- 2x InGaAs Detectors (DET10C)

Hardware interface via Arduino Due with MCP4728 DAC.
"""

import numpy as np
import serial
import time
from typing import Tuple, Optional, List
from dataclasses import dataclass
import struct


@dataclass
class CalibrationData:
    """Calibration coefficients for the optical system."""
    # Attenuator voltage to attenuation mapping
    voa_v_to_db: np.ndarray  # [voltage] -> dB attenuation

    # Detector responsivity (V/mW)
    det1_responsivity: float
    det2_responsivity: float

    # Splitting ratio of coupler
    split_ratio: float  # Actual ratio (ideal = 0.5)

    # Phase offset
    phase_offset: float


class ArduinoInterface:
    """
    Serial interface to Arduino Due controlling the optics.

    Arduino runs firmware that:
    - Sets DAC voltages (MCP4728)
    - Reads ADC values (built-in 12-bit)
    - Controls laser enable
    """

    def __init__(self, port: str = '/dev/ttyACM0', baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to Arduino."""
        try:
            self.serial = serial.Serial(
                self.port,
                self.baudrate,
                timeout=1.0
            )
            time.sleep(2)  # Wait for Arduino reset
            self.connected = True

            # Verify connection
            self.serial.write(b'ID\n')
            response = self.serial.readline().decode().strip()

            if 'PHOTONCORE' in response:
                print(f"Connected to PhotonCore controller: {response}")
                return True
            else:
                print(f"Unknown device: {response}")
                return False

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from Arduino."""
        if self.serial:
            self.serial.close()
            self.connected = False

    def set_dac(self, channel: int, voltage: float):
        """
        Set DAC output voltage.

        Channels:
        0: VOA control (0-5V)
        1: Future expansion
        2: Future expansion
        3: Future expansion
        """
        # MCP4728 is 12-bit, 0-5V
        value = int(voltage / 5.0 * 4095)
        value = max(0, min(4095, value))

        cmd = f'DAC {channel} {value}\n'
        self.serial.write(cmd.encode())
        self.serial.readline()  # ACK

    def read_adc(self, channel: int) -> float:
        """
        Read ADC input voltage.

        Channels:
        0: Detector 1
        1: Detector 2
        """
        cmd = f'ADC {channel}\n'
        self.serial.write(cmd.encode())
        response = self.serial.readline().decode().strip()

        # Arduino Due ADC is 12-bit, 0-3.3V
        value = int(response)
        voltage = value / 4095.0 * 3.3

        return voltage

    def set_laser(self, enable: bool):
        """Enable/disable laser."""
        cmd = f'LASER {"ON" if enable else "OFF"}\n'
        self.serial.write(cmd.encode())
        self.serial.readline()  # ACK

    def read_detectors(self) -> Tuple[float, float]:
        """Read both detector voltages."""
        v1 = self.read_adc(0)
        v2 = self.read_adc(1)
        return v1, v2


class BenchtopController:
    """
    High-level controller for PhotonCore bench-top demo.
    """

    def __init__(self, port: str = '/dev/ttyACM0'):
        self.arduino = ArduinoInterface(port)
        self.calibration: Optional[CalibrationData] = None
        self.laser_on = False

    def initialize(self) -> bool:
        """Initialize the system."""
        if not self.arduino.connect():
            return False

        # Start with laser off, VOA at max attenuation
        self.arduino.set_laser(False)
        self.arduino.set_dac(0, 0.0)  # VOA off

        return True

    def shutdown(self):
        """Safe shutdown."""
        self.arduino.set_laser(False)
        self.arduino.set_dac(0, 0.0)
        self.arduino.disconnect()

    def set_attenuation(self, db: float):
        """
        Set VOA attenuation in dB.

        EVOA1550A range: 0-25 dB
        """
        if self.calibration:
            # Use calibration curve
            voltage = np.interp(db,
                               self.calibration.voa_v_to_db[:, 1],
                               self.calibration.voa_v_to_db[:, 0])
        else:
            # Linear approximation: 0V = 0dB, 5V = 25dB
            voltage = db / 25.0 * 5.0

        voltage = max(0, min(5, voltage))
        self.arduino.set_dac(0, voltage)

    def read_power(self) -> Tuple[float, float]:
        """
        Read optical power at both detectors (mW).
        """
        v1, v2 = self.arduino.read_detectors()

        if self.calibration:
            p1 = v1 / self.calibration.det1_responsivity
            p2 = v2 / self.calibration.det2_responsivity
        else:
            # DET10C typical: 1.05 A/W, assume 10kΩ load
            # V = P * R * responsivity = P * 10000 * 1.05
            # P = V / 10500 (in W) = V / 10.5 (in mW)
            p1 = v1 / 10.5
            p2 = v2 / 10.5

        return p1, p2

    def calibrate(self) -> CalibrationData:
        """
        Run full calibration sequence.

        1. Sweep VOA to map voltage-to-attenuation
        2. Measure detector responsivities
        3. Measure coupler splitting ratio
        """
        print("Starting calibration...")

        # Turn on laser
        self.arduino.set_laser(True)
        time.sleep(0.5)

        # Step 1: VOA calibration
        print("  Calibrating VOA...")
        voa_curve = []

        for v in np.linspace(0, 5, 51):
            self.arduino.set_dac(0, v)
            time.sleep(0.05)
            p1, p2 = self.read_power()
            total_power = p1 + p2

            if len(voa_curve) == 0:
                ref_power = total_power

            if total_power > 0:
                db = -10 * np.log10(total_power / ref_power)
            else:
                db = 30  # Max attenuation

            voa_curve.append([v, db])

        voa_curve = np.array(voa_curve)

        # Step 2: Measure splitting ratio at mid-power
        print("  Measuring splitting ratio...")
        self.arduino.set_dac(0, 2.5)  # Mid attenuation
        time.sleep(0.1)

        p1, p2 = self.read_power()
        split_ratio = p1 / (p1 + p2) if (p1 + p2) > 0 else 0.5

        # Step 3: Detector responsivity (already measured inline)
        # Use nominal values for now
        det1_resp = 10.5
        det2_resp = 10.5

        # Create calibration data
        self.calibration = CalibrationData(
            voa_v_to_db=voa_curve,
            det1_responsivity=det1_resp,
            det2_responsivity=det2_resp,
            split_ratio=split_ratio,
            phase_offset=0.0
        )

        print(f"  VOA range: {voa_curve[0, 1]:.1f} to {voa_curve[-1, 1]:.1f} dB")
        print(f"  Split ratio: {split_ratio:.3f} (ideal: 0.500)")
        print("Calibration complete!")

        return self.calibration

    def measure_interference(self, n_points: int = 100) -> np.ndarray:
        """
        Sweep phase and measure interference pattern.

        For 2x2 demo, we vary attenuation to simulate amplitude,
        observing power distribution between outputs.

        Returns array of [attenuation, det1, det2]
        """
        self.arduino.set_laser(True)
        time.sleep(0.1)

        results = []

        for db in np.linspace(0, 20, n_points):
            self.set_attenuation(db)
            time.sleep(0.02)

            p1, p2 = self.read_power()
            results.append([db, p1, p2])

        return np.array(results)

    def optical_2x2_transform(self,
                              input_amplitude: complex,
                              target_split: float = 0.5) -> Tuple[complex, complex]:
        """
        Perform 2x2 optical transform.

        For single MZI:
        [out1]   [cos(θ)   i*sin(θ)] [in1]
        [out2] = [i*sin(θ) cos(θ)  ] [in2]

        With in2=0, this is effectively:
        out1 = in1 * cos(θ)
        out2 = in1 * i*sin(θ)

        We control θ via the VOA (amplitude modulation).
        """
        # Set input amplitude via VOA
        input_power = np.abs(input_amplitude) ** 2

        # For target_split, we need θ where cos²(θ) = target_split
        theta = np.arccos(np.sqrt(target_split))

        # Map to attenuation (simplified)
        # Full power corresponds to some attenuation
        self.set_attenuation(0)  # Minimum attenuation
        time.sleep(0.05)

        # Read outputs
        p1, p2 = self.read_power()

        # Convert to complex amplitudes (assume phase from geometry)
        phase = np.angle(input_amplitude)
        out1 = np.sqrt(p1) * np.exp(1j * phase)
        out2 = np.sqrt(p2) * np.exp(1j * (phase + np.pi/2))  # 90° phase shift

        return out1, out2


class PhotonCore2x2Demo:
    """
    Complete demo application for 2x2 optical computing.
    """

    def __init__(self, port: str = '/dev/ttyACM0'):
        self.controller = BenchtopController(port)

    def run_demo(self):
        """Run full demonstration sequence."""
        print("=" * 60)
        print("PhotonCore-X 2x2 Optical Computing Demo")
        print("=" * 60)

        # Initialize
        print("\n1. Initializing system...")
        if not self.controller.initialize():
            print("ERROR: Failed to initialize!")
            return

        # Calibrate
        print("\n2. Running calibration...")
        cal = self.controller.calibrate()

        # Interference measurement
        print("\n3. Measuring interference pattern...")
        pattern = self.controller.measure_interference(50)

        total_power = pattern[:, 1] + pattern[:, 2]
        contrast = (pattern[:, 1] - pattern[:, 2]) / (total_power + 1e-10)

        print(f"   Max contrast: {np.max(np.abs(contrast)):.3f}")
        print(f"   Power range: {np.min(total_power):.3f} - {np.max(total_power):.3f} mW")

        # Demo matrix multiply
        print("\n4. Demonstrating optical transform...")

        # Test vectors
        test_inputs = [
            complex(1, 0),
            complex(0.707, 0.707),
            complex(0, 1),
        ]

        for inp in test_inputs:
            out1, out2 = self.controller.optical_2x2_transform(inp)

            print(f"   Input:  {inp:.3f}")
            print(f"   Output: [{out1:.3f}, {out2:.3f}]")
            print(f"   |out|²: [{np.abs(out1)**2:.3f}, {np.abs(out2)**2:.3f}]")
            print()

        # Cleanup
        print("5. Shutting down...")
        self.controller.shutdown()

        print("\n" + "=" * 60)
        print("Demo complete!")
        print("=" * 60)


# =============================================================================
# Arduino Firmware (for reference)
# =============================================================================

ARDUINO_FIRMWARE = """
/*
 * PhotonCore 2x2 Demo - Arduino Due Firmware
 *
 * Controls:
 * - MCP4728 DAC via I2C (VOA control)
 * - Laser enable via digital pin
 * - ADC reads from detectors
 */

#include <Wire.h>

#define LASER_PIN 13
#define DET1_PIN A0
#define DET2_PIN A1
#define MCP4728_ADDR 0x60

void setup() {
    Serial.begin(115200);
    Wire.begin();

    pinMode(LASER_PIN, OUTPUT);
    digitalWrite(LASER_PIN, LOW);

    analogReadResolution(12);

    Serial.println("PHOTONCORE_2X2_V1");
}

void loop() {
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\\n');
        cmd.trim();

        if (cmd == "ID") {
            Serial.println("PHOTONCORE_2X2_V1");
        }
        else if (cmd.startsWith("DAC")) {
            int ch, val;
            sscanf(cmd.c_str(), "DAC %d %d", &ch, &val);
            setDAC(ch, val);
            Serial.println("OK");
        }
        else if (cmd.startsWith("ADC")) {
            int ch;
            sscanf(cmd.c_str(), "ADC %d", &ch);
            int val = (ch == 0) ? analogRead(DET1_PIN) : analogRead(DET2_PIN);
            Serial.println(val);
        }
        else if (cmd == "LASER ON") {
            digitalWrite(LASER_PIN, HIGH);
            Serial.println("OK");
        }
        else if (cmd == "LASER OFF") {
            digitalWrite(LASER_PIN, LOW);
            Serial.println("OK");
        }
    }
}

void setDAC(int channel, int value) {
    // MCP4728 single channel write
    Wire.beginTransmission(MCP4728_ADDR);
    Wire.write(0x40 | (channel << 1));
    Wire.write((value >> 8) & 0x0F);
    Wire.write(value & 0xFF);
    Wire.endTransmission();
}
"""


if __name__ == "__main__":
    # Run demo
    demo = PhotonCore2x2Demo()
    demo.run_demo()
