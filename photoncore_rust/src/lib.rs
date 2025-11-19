//! PhotonCore-X Photonic AI Accelerator Simulation
//!
//! High-performance Rust implementation of photonic quantum-classical
//! hybrid computing primitives.

pub mod mzi;
pub mod clements;
pub mod optical_mvm;
pub mod calibration;
pub mod wdm;

pub use mzi::{MachZehnderInterferometer, MZIMesh};
pub use clements::{ClementsDecomposition, random_unitary};
pub use optical_mvm::{OpticalMatrixUnit, OpticalNonlinearity};
pub use calibration::{CalibrationSystem, AutoCalibrator};
pub use wdm::{WDMSystem, KerrComb};
