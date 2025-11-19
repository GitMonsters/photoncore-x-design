//! Wavelength Division Multiplexing (WDM) System

/// WDM system for wavelength-parallel optical computing.
pub struct WDMSystem {
    pub n_channels: usize,
    pub center_wavelength: f64,
    pub channel_spacing: f64,
    pub wavelengths: Vec<f64>,
    pub powers: Vec<f64>,
}

impl WDMSystem {
    /// Create new WDM system.
    pub fn new(n_channels: usize, center_wavelength_nm: f64, channel_spacing_ghz: f64) -> Self {
        let c = 299792458.0; // m/s
        let center_freq = c / (center_wavelength_nm * 1e-9);

        let frequencies: Vec<f64> = (0..n_channels)
            .map(|i| center_freq + (i as f64 - n_channels as f64 / 2.0) * channel_spacing_ghz * 1e9)
            .collect();

        let wavelengths: Vec<f64> = frequencies.iter().map(|f| c / f * 1e9).collect();

        let powers = vec![1.0; n_channels];

        Self {
            n_channels,
            center_wavelength: center_wavelength_nm,
            channel_spacing: channel_spacing_ghz,
            wavelengths,
            powers,
        }
    }

    /// Get channel information.
    pub fn get_channel_info(&self, idx: usize) -> (f64, f64, f64) {
        let wavelength = self.wavelengths[idx];
        let c = 299792458.0;
        let frequency = c / (wavelength * 1e-9) / 1e12; // THz
        let power = self.powers[idx];
        (wavelength, frequency, power)
    }
}

/// Kerr frequency comb light source.
pub struct KerrComb {
    pub n_lines: usize,
    pub pump_wavelength: f64,
    pub fsr: f64,
    pub pump_power: f64,
    pub wavelengths: Vec<f64>,
    pub powers: Vec<f64>,
}

impl KerrComb {
    /// Create new Kerr comb.
    pub fn new(n_lines: usize, pump_wavelength_nm: f64, fsr_ghz: f64, pump_power_mw: f64) -> Self {
        let c = 299792458.0;
        let pump_freq = c / (pump_wavelength_nm * 1e-9);

        let frequencies: Vec<f64> = (0..n_lines)
            .map(|i| pump_freq + (i as f64 - n_lines as f64 / 2.0) * fsr_ghz * 1e9)
            .collect();

        let wavelengths: Vec<f64> = frequencies.iter().map(|f| c / f * 1e9).collect();

        // Sech² envelope for soliton combs
        let powers: Vec<f64> = (0..n_lines)
            .map(|i| {
                let x = (i as f64 - n_lines as f64 / 2.0) / (n_lines as f64 / 4.0);
                let envelope = 1.0 / x.cosh().powi(2);
                pump_power_mw * envelope / n_lines as f64
            })
            .collect();

        Self {
            n_lines,
            pump_wavelength: pump_wavelength_nm,
            fsr: fsr_ghz,
            pump_power: pump_power_mw,
            wavelengths,
            powers,
        }
    }

    /// Get total output power.
    pub fn total_power(&self) -> f64 {
        self.powers.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wdm_system() {
        let wdm = WDMSystem::new(16, 1550.0, 50.0);
        assert_eq!(wdm.n_channels, 16);
        assert_eq!(wdm.wavelengths.len(), 16);
    }

    #[test]
    fn test_kerr_comb() {
        let comb = KerrComb::new(64, 1550.0, 100.0, 100.0);
        assert_eq!(comb.n_lines, 64);
        assert!(comb.total_power() > 0.0);
    }
}
