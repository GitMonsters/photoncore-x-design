//! Mach-Zehnder Interferometer (MZI) Simulation
//!
//! Core building block for programmable photonic circuits.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Single MZI unit with two phase shifters (theta, phi).
///
/// Implements the transformation:
/// T(θ, φ) = [[e^(iφ)cos(θ), -sin(θ)],
///            [e^(iφ)sin(θ),  cos(θ)]]
#[derive(Clone, Debug)]
pub struct MachZehnderInterferometer {
    pub theta: f64,
    pub phi: f64,
    matrix: Array2<Complex64>,
}

impl MachZehnderInterferometer {
    /// Create new MZI with given phases.
    pub fn new(theta: f64, phi: f64) -> Self {
        let matrix = Self::compute_matrix(theta, phi);
        Self { theta, phi, matrix }
    }

    /// Compute transfer matrix from phases.
    #[inline]
    fn compute_matrix(theta: f64, phi: f64) -> Array2<Complex64> {
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let exp_phi = Complex64::from_polar(1.0, phi);

        Array2::from_shape_vec(
            (2, 2),
            vec![
                exp_phi * cos_t,
                Complex64::new(-sin_t, 0.0),
                exp_phi * sin_t,
                Complex64::new(cos_t, 0.0),
            ],
        )
        .unwrap()
    }

    /// Update phase settings.
    pub fn set_phases(&mut self, theta: f64, phi: f64) {
        self.theta = theta;
        self.phi = phi;
        self.matrix = Self::compute_matrix(theta, phi);
    }

    /// Apply MZI transformation to input optical fields.
    #[inline]
    pub fn forward(&self, inputs: &Array1<Complex64>) -> Array1<Complex64> {
        self.matrix.dot(inputs)
    }

    /// Get the transfer matrix.
    pub fn matrix(&self) -> &Array2<Complex64> {
        &self.matrix
    }
}

impl Default for MachZehnderInterferometer {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

/// Rectangular mesh of MZIs implementing arbitrary NxN unitary matrix.
///
/// Uses the Clements decomposition architecture with N(N-1)/2 MZIs.
#[derive(Clone)]
pub struct MZIMesh {
    pub n_ports: usize,
    pub n_mzis: usize,
    mzis: Vec<MachZehnderInterferometer>,
    mzi_positions: Vec<(usize, usize)>, // (layer, position)

    // Physical parameters
    insertion_loss: f64,
    phase_noise_std: f64,
    crosstalk: f64,

    // Phase arrays
    pub thetas: Vec<f64>,
    pub phis: Vec<f64>,
    pub output_phases: Vec<f64>,
}

impl MZIMesh {
    /// Create new MZI mesh.
    pub fn new(
        n_ports: usize,
        insertion_loss_db: f64,
        phase_noise_std: f64,
        crosstalk_db: f64,
    ) -> Self {
        let n_mzis = n_ports * (n_ports - 1) / 2;

        let insertion_loss = 10_f64.powf(-insertion_loss_db / 10.0);
        let crosstalk = 10_f64.powf(crosstalk_db / 10.0);

        let mut mzis = Vec::with_capacity(n_mzis);
        let mut mzi_positions = Vec::with_capacity(n_mzis);

        // Build mesh topology (Clements architecture)
        let mut mzi_idx = 0;
        for layer in 0..(2 * n_ports - 3) {
            let start = if layer % 2 == 0 { 0 } else { 1 };

            let mut pos = start;
            while pos < n_ports - 1 {
                if mzi_idx >= n_mzis {
                    break;
                }
                mzis.push(MachZehnderInterferometer::default());
                mzi_positions.push((layer, pos));
                mzi_idx += 1;
                pos += 2;
            }
        }

        Self {
            n_ports,
            n_mzis,
            mzis,
            mzi_positions,
            insertion_loss,
            phase_noise_std,
            crosstalk,
            thetas: vec![0.0; n_mzis],
            phis: vec![0.0; n_mzis],
            output_phases: vec![0.0; n_ports],
        }
    }

    /// Set all phase values in the mesh.
    pub fn set_phases(&mut self, thetas: &[f64], phis: &[f64], output_phases: &[f64]) {
        assert_eq!(thetas.len(), self.n_mzis);
        assert_eq!(phis.len(), self.n_mzis);
        assert_eq!(output_phases.len(), self.n_ports);

        self.thetas = thetas.to_vec();
        self.phis = phis.to_vec();
        self.output_phases = output_phases.to_vec();

        for (i, mzi) in self.mzis.iter_mut().enumerate() {
            mzi.set_phases(thetas[i], phis[i]);
        }
    }

    /// Propagate optical fields through the mesh.
    pub fn forward(&self, inputs: &Array1<Complex64>, add_noise: bool) -> Array1<Complex64> {
        let mut state = inputs.clone();
        let n = self.n_ports;

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, self.phase_noise_std).unwrap();

        let mut mzi_idx = 0;

        // Process each layer
        for layer in 0..(2 * n - 3) {
            let start = if layer % 2 == 0 { 0 } else { 1 };

            let mut pos = start;
            while pos < n - 1 {
                if mzi_idx >= self.n_mzis {
                    break;
                }

                let mzi = &self.mzis[mzi_idx];

                // Get MZI matrix (optionally with noise)
                let matrix = if add_noise {
                    let noise_theta = normal.sample(&mut rng);
                    let noise_phi = normal.sample(&mut rng);
                    let mut noisy_matrix = MachZehnderInterferometer::compute_matrix(
                        mzi.theta + noise_theta,
                        mzi.phi + noise_phi,
                    );
                    // Apply insertion loss
                    noisy_matrix *= Complex64::new(self.insertion_loss.sqrt(), 0.0);
                    noisy_matrix
                } else {
                    mzi.matrix().clone()
                };

                // Apply MZI to ports (pos, pos+1)
                let input_vec = Array1::from_vec(vec![state[pos], state[pos + 1]]);
                let output_vec = matrix.dot(&input_vec);
                state[pos] = output_vec[0];
                state[pos + 1] = output_vec[1];

                mzi_idx += 1;
                pos += 2;
            }
        }

        // Apply output phase shifters
        for i in 0..n {
            state[i] *= Complex64::from_polar(1.0, self.output_phases[i]);
        }

        state
    }

    /// Compute the full transfer matrix of the mesh.
    pub fn get_matrix(&self, add_noise: bool) -> Array2<Complex64> {
        let mut matrix = Array2::zeros((self.n_ports, self.n_ports));

        for i in 0..self.n_ports {
            let mut basis = Array1::zeros(self.n_ports);
            basis[i] = Complex64::new(1.0, 0.0);
            let output = self.forward(&basis, add_noise);
            for j in 0..self.n_ports {
                matrix[[j, i]] = output[j];
            }
        }

        matrix
    }

    /// Check if current configuration implements a unitary matrix.
    pub fn is_unitary(&self, tolerance: f64) -> bool {
        let u = self.get_matrix(false);
        let u_h = u.t().mapv(|x| x.conj());
        let product = u.dot(&u_h);

        for i in 0..self.n_ports {
            for j in 0..self.n_ports {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (product[[i, j]].norm() - expected).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }
}

/// Create a mesh configured as identity.
pub fn create_identity_mesh(n_ports: usize) -> MZIMesh {
    let mut mesh = MZIMesh::new(n_ports, 0.1, 0.0, -40.0);
    mesh.set_phases(
        &vec![0.0; mesh.n_mzis],
        &vec![0.0; mesh.n_mzis],
        &vec![0.0; n_ports],
    );
    mesh
}

/// Create a mesh with random unitary configuration.
pub fn create_random_unitary_mesh(n_ports: usize) -> MZIMesh {
    let mut mesh = MZIMesh::new(n_ports, 0.1, 0.0, -40.0);
    let mut rng = rand::thread_rng();

    let thetas: Vec<f64> = (0..mesh.n_mzis)
        .map(|_| rng.gen_range(0.0..PI / 2.0))
        .collect();
    let phis: Vec<f64> = (0..mesh.n_mzis)
        .map(|_| rng.gen_range(0.0..2.0 * PI))
        .collect();
    let output: Vec<f64> = (0..n_ports)
        .map(|_| rng.gen_range(0.0..2.0 * PI))
        .collect();

    mesh.set_phases(&thetas, &phis, &output);
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mzi_identity() {
        let mzi = MachZehnderInterferometer::new(0.0, 0.0);
        let input = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        let output = mzi.forward(&input);

        assert!((output[0].norm() - 1.0).abs() < 1e-10);
        assert!(output[1].norm() < 1e-10);
    }

    #[test]
    fn test_mesh_unitarity() {
        let mesh = create_random_unitary_mesh(8);
        assert!(mesh.is_unitary(1e-6));
    }

    #[test]
    fn test_mesh_forward() {
        let mesh = create_random_unitary_mesh(4);

        let mut input = Array1::zeros(4);
        input[0] = Complex64::new(1.0, 0.0);

        let output = mesh.forward(&input, false);
        let norm: f64 = output.iter().map(|x| x.norm_sqr()).sum();

        // Should preserve norm (unitary)
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
