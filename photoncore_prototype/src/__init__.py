"""
PhotonCore-X Prototype Simulation Framework

A comprehensive simulation and prototyping toolkit for photonic
quantum-classical hybrid AI accelerators.
"""

from mzi import MachZehnderInterferometer, MZIMesh
from clements import ClementsDecomposition, clements_decompose, clements_reconstruct
from optical_mvm import OpticalMatrixUnit, OpticalNonlinearity
from calibration import CalibrationSystem, AutoCalibrator
from photoncore import PhotonCoreSimulator, PhotonCoreSDK
from wdm import WDMSystem, KerrComb

__version__ = "0.1.0"
__author__ = "PhotonCore Team"

__all__ = [
    'MachZehnderInterferometer',
    'MZIMesh',
    'ClementsDecomposition',
    'clements_decompose',
    'clements_reconstruct',
    'OpticalMatrixUnit',
    'OpticalNonlinearity',
    'CalibrationSystem',
    'AutoCalibrator',
    'PhotonCoreSimulator',
    'PhotonCoreSDK',
    'WDMSystem',
    'KerrComb',
]
