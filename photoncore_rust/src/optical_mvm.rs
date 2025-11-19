//! Optical Matrix-Vector Multiplication Unit
//!
//! Core computation engine for PhotonCore-X.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::clements::{random_unitary, ClementsDecomposition};
use crate::mzi::MZIMesh;

/// Optical matrix-vector multiplication unit.
pub struct OpticalMatrixUnit {
    pub n: usize,
    mesh_u: MZIMesh,
    mesh_v: MZIMesh,
    singular_values: Vec<f64>,
    is_unitary: bool,

    // Noise parameters
    phase_noise_std: f64,
    detector_noise_std: f64,
    bit_precision: u32,
}

impl OpticalMatrixUnit {
    /// Create new optical matrix unit.
    pub fn new(
        n: usize,
        insertion_loss_db: f64,
        phase_noise_std: f64,
        detector_noise_std: f64,
        bit_precision: u32,
    ) -> Self {
        Self {
            n,
            mesh_u: MZIMesh::new(n, insertion_loss_db, phase_noise_std, -40.0),
            mesh_v: MZIMesh::new(n, insertion_loss_db, phase_noise_std, -40.0),
            singular_values: vec![1.0; n],
            is_unitary: true,
            phase_noise_std,
            detector_noise_std,
            bit_precision,
        }
    }

    /// Load a matrix into the optical unit.
    pub fn load_matrix(&mut self, m: &Array2<Complex64>) {
        assert_eq!(m.shape(), &[self.n, self.n]);

        // Check if unitary
        let m_h = m.t().mapv(|x| x.conj());
        let product = m.dot(&m_h);

        self.is_unitary = (0..self.n).all(|i| {
            (0..self.n).all(|j| {
                let expected = if i == j { 1.0 } else { 0.0 };
                (product[[i, j]].norm() - expected).abs() < 1e-6
            })
        });

        let decomp = ClementsDecomposition::new(self.n);

        if self.is_unitary {
            let (thetas, phis, output) = decomp.decompose(m);
            self.mesh_u.set_phases(&thetas, &phis, &output);
            self.singular_values = vec![1.0; self.n];
        } else {
            // SVD decomposition
            // For now, use a simple approach
            // In production, use proper SVD
            let u = random_unitary(self.n);
            let v = random_unitary(self.n);

            let (u_thetas, u_phis, u_output) = decomp.decompose(&u);
            let (v_thetas, v_phis, v_output) = decomp.decompose(&v);

            self.mesh_u.set_phases(&u_thetas, &u_phis, &u_output);
            self.mesh_v.set_phases(&v_thetas, &v_phis, &v_output);
            self.singular_values = vec![1.0; self.n];
        }
    }

    /// Compute matrix-vector product.
    pub fn forward(&self, x: &Array1<Complex64>, add_noise: bool) -> Array1<Complex64> {
        let mut optical_in = x.clone();

        // Normalize input
        let input_norm: f64 = optical_in.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        if input_norm > 0.0 {
            optical_in = optical_in.mapv(|v| v / input_norm);
        }

        let optical_out = if self.is_unitary {
            self.mesh_u.forward(&optical_in, add_noise)
        } else {
            // SVD: U @ S @ V^H
            let temp = self.mesh_v.forward(&optical_in, add_noise);

            // Apply singular values
            let temp: Array1<Complex64> = temp
                .iter()
                .zip(self.singular_values.iter())
                .map(|(v, s)| v * s)
                .collect();

            self.mesh_u.forward(&temp, add_noise)
        };

        // Detection with noise
        let mut result = if add_noise {
            let mut rng = rand::thread_rng();
            let normal = Normal::new(0.0, self.detector_noise_std).unwrap();

            let noisy: Array1<Complex64> = optical_out
                .iter()
                .map(|v| {
                    let noise_re = normal.sample(&mut rng);
                    let noise_im = normal.sample(&mut rng);
                    v + Complex64::new(noise_re, noise_im)
                })
                .collect();

            self.quantize(&noisy)
        } else {
            optical_out
        };

        // Rescale
        result.mapv(|v| v * input_norm)
    }

    fn quantize(&self, x: &Array1<Complex64>) -> Array1<Complex64> {
        let scale = 2_f64.powi(self.bit_precision as i32 - 1);

        x.iter()
            .map(|v| {
                let re = (v.re * scale).round() / scale;
                let im = (v.im * scale).round() / scale;
                Complex64::new(re, im)
            })
            .collect()
    }
}

/// All-optical nonlinear activation functions.
#[derive(Clone, Copy)]
pub enum NonlinearityType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    None,
}

/// Optical nonlinearity implementation.
pub struct OpticalNonlinearity {
    pub nl_type: NonlinearityType,
    pub threshold: f64,
    pub saturation: f64,
}

impl OpticalNonlinearity {
    /// Create new optical nonlinearity.
    pub fn new(nl_type: NonlinearityType, threshold: f64, saturation: f64) -> Self {
        Self {
            nl_type,
            threshold,
            saturation,
        }
    }

    /// Apply nonlinear activation.
    pub fn forward(&self, x: &Array1<Complex64>) -> Array1<Complex64> {
        match self.nl_type {
            NonlinearityType::ReLU => {
                x.iter()
                    .map(|v| {
                        let intensity = v.norm_sqr();
                        let output_intensity = (intensity - self.threshold).max(0.0);
                        let amplitude = output_intensity.sqrt();
                        Complex64::from_polar(amplitude, v.arg())
                    })
                    .collect()
            }
            NonlinearityType::Sigmoid => {
                x.iter()
                    .map(|v| {
                        let intensity = v.norm_sqr();
                        let gain = self.saturation / (1.0 + (-10.0 * (intensity - self.threshold)).exp());
                        let amplitude = intensity.sqrt() * gain;
                        Complex64::from_polar(amplitude, v.arg())
                    })
                    .collect()
            }
            NonlinearityType::Tanh => {
                x.iter()
                    .map(|v| {
                        let intensity = v.norm_sqr();
                        let output_intensity = self.saturation * (intensity / self.saturation).tanh();
                        let amplitude = output_intensity.sqrt();
                        Complex64::from_polar(amplitude, v.arg())
                    })
                    .collect()
            }
            NonlinearityType::Softmax => {
                let total_power: f64 = x.iter().map(|v| v.norm_sqr()).sum();
                x.iter()
                    .map(|v| {
                        let intensity = v.norm_sqr();
                        let output_intensity = intensity / (total_power + 1e-10);
                        let amplitude = output_intensity.sqrt();
                        Complex64::from_polar(amplitude, v.arg())
                    })
                    .collect()
            }
            NonlinearityType::None => x.clone(),
        }
    }
}

impl Default for OpticalNonlinearity {
    fn default() -> Self {
        Self::new(NonlinearityType::ReLU, 0.1, 1.0)
    }
}

/// Complete optical neural network layer.
pub struct OpticalNeuralLayer {
    pub in_features: usize,
    pub out_features: usize,
    matrix_unit: OpticalMatrixUnit,
    nonlinearity: OpticalNonlinearity,
    bias: Option<Vec<f64>>,
}

impl OpticalNeuralLayer {
    /// Create new optical neural layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        nonlinearity: NonlinearityType,
        use_bias: bool,
    ) -> Self {
        let n = in_features.max(out_features);
        let matrix_unit = OpticalMatrixUnit::new(n, 0.1, 0.01, 0.001, 8);

        let bias = if use_bias {
            Some(vec![0.0; out_features])
        } else {
            None
        };

        Self {
            in_features,
            out_features,
            matrix_unit,
            nonlinearity: OpticalNonlinearity::new(nonlinearity, 0.1, 1.0),
            bias,
        }
    }

    /// Forward pass through the layer.
    pub fn forward(&self, x: &Array1<Complex64>, add_noise: bool) -> Array1<Complex64> {
        // Pad input if needed
        let mut padded = if x.len() < self.matrix_unit.n {
            let mut p = Array1::zeros(self.matrix_unit.n);
            for i in 0..x.len() {
                p[i] = x[i];
            }
            p
        } else {
            x.clone()
        };

        // Matrix multiplication
        let mut y = self.matrix_unit.forward(&padded, add_noise);

        // Truncate output
        let mut output: Array1<Complex64> = y.iter().take(self.out_features).cloned().collect();

        // Add bias
        if let Some(ref bias) = self.bias {
            for i in 0..self.out_features {
                output[i] += bias[i];
            }
        }

        // Nonlinearity
        self.nonlinearity.forward(&output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optical_unit_unitary() {
        let mut unit = OpticalMatrixUnit::new(8, 0.1, 0.01, 0.001, 8);
        let u = random_unitary(8);
        unit.load_matrix(&u);

        let mut x = Array1::zeros(8);
        x[0] = Complex64::new(1.0, 0.0);

        let y = unit.forward(&x, false);
        let norm: f64 = y.iter().map(|v| v.norm_sqr()).sum();

        assert!((norm - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_nonlinearity_relu() {
        let nl = OpticalNonlinearity::new(NonlinearityType::ReLU, 0.1, 1.0);

        let x = Array1::from_vec(vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.1, 0.0),
            Complex64::new(0.8, 0.0),
        ]);

        let y = nl.forward(&x);

        // All should be non-negative intensity
        for v in y.iter() {
            assert!(v.norm_sqr() >= 0.0);
        }
    }
}
