//! Calibration System for Photonic Circuits

use serde::{Deserialize, Serialize};
use std::fs;

/// Calibration system for MZI mesh.
#[derive(Clone, Serialize, Deserialize)]
pub struct CalibrationSystem {
    pub n_ports: usize,
    pub n_mzis: usize,
    pub theta_offsets: Vec<f64>,
    pub phi_offsets: Vec<f64>,
    pub theta_scales: Vec<f64>,
    pub phi_scales: Vec<f64>,
    pub output_offsets: Vec<f64>,
}

impl CalibrationSystem {
    /// Create new calibration system.
    pub fn new(n_ports: usize, n_mzis: usize) -> Self {
        Self {
            n_ports,
            n_mzis,
            theta_offsets: vec![0.0; n_mzis],
            phi_offsets: vec![0.0; n_mzis],
            theta_scales: vec![1.0; n_mzis],
            phi_scales: vec![1.0; n_mzis],
            output_offsets: vec![0.0; n_ports],
        }
    }

    /// Apply calibration corrections to phase settings.
    pub fn correct_phases(
        &self,
        thetas: &[f64],
        phis: &[f64],
        output_phases: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let corrected_thetas: Vec<f64> = thetas
            .iter()
            .zip(self.theta_offsets.iter())
            .zip(self.theta_scales.iter())
            .map(|((t, off), scale)| (t - off) / scale)
            .collect();

        let corrected_phis: Vec<f64> = phis
            .iter()
            .zip(self.phi_offsets.iter())
            .zip(self.phi_scales.iter())
            .map(|((p, off), scale)| (p - off) / scale)
            .collect();

        let corrected_output: Vec<f64> = output_phases
            .iter()
            .zip(self.output_offsets.iter())
            .map(|(p, off)| p - off)
            .collect();

        (corrected_thetas, corrected_phis, corrected_output)
    }

    /// Save calibration to file.
    pub fn save(&self, filepath: &str) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(filepath, json)
    }

    /// Load calibration from file.
    pub fn load(filepath: &str) -> Result<Self, std::io::Error> {
        let json = fs::read_to_string(filepath)?;
        let cal: Self = serde_json::from_str(&json)?;
        Ok(cal)
    }
}

/// Auto-calibrator for MZI mesh.
pub struct AutoCalibrator {
    pub n_ports: usize,
    pub n_mzis: usize,
}

impl AutoCalibrator {
    /// Create new auto-calibrator.
    pub fn new(n_ports: usize, n_mzis: usize) -> Self {
        Self { n_ports, n_mzis }
    }

    /// Calibrate single MZI (placeholder).
    pub fn calibrate_single_mzi(&self, _mzi_idx: usize) -> (f64, f64) {
        // In production: sweep phases and measure response
        (0.0, 1.0) // offset, scale
    }

    /// Calibrate full mesh.
    pub fn calibrate_full_mesh(&self) -> CalibrationSystem {
        let mut cal = CalibrationSystem::new(self.n_ports, self.n_mzis);

        for i in 0..self.n_mzis {
            let (offset, scale) = self.calibrate_single_mzi(i);
            cal.theta_offsets[i] = offset;
            cal.theta_scales[i] = scale;
        }

        cal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_save_load() {
        let cal = CalibrationSystem::new(8, 28);
        let path = "/tmp/test_cal.json";

        cal.save(path).unwrap();
        let loaded = CalibrationSystem::load(path).unwrap();

        assert_eq!(cal.n_ports, loaded.n_ports);
        assert_eq!(cal.n_mzis, loaded.n_mzis);
    }

    #[test]
    fn test_phase_correction() {
        let mut cal = CalibrationSystem::new(4, 6);
        cal.theta_offsets[0] = 0.1;
        cal.theta_scales[0] = 1.1;

        let thetas = vec![0.5, 0.3, 0.2, 0.1, 0.4, 0.6];
        let phis = vec![0.0; 6];
        let output = vec![0.0; 4];

        let (corr_thetas, _, _) = cal.correct_phases(&thetas, &phis, &output);

        // First theta should be corrected
        let expected = (0.5 - 0.1) / 1.1;
        assert!((corr_thetas[0] - expected).abs() < 1e-10);
    }
}
