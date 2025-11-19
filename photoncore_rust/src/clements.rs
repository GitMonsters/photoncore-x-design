//! Clements Decomposition for Unitary Matrices
//!
//! Decomposes any NxN unitary matrix into MZI mesh parameters.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rand::Rng;
use std::f64::consts::PI;

use crate::mzi::MZIMesh;

/// Clements decomposition for unitary matrices.
pub struct ClementsDecomposition {
    pub n: usize,
    pub n_mzis: usize,
}

impl ClementsDecomposition {
    /// Create new decomposition for NxN matrices.
    pub fn new(n: usize) -> Self {
        let n_mzis = n * (n - 1) / 2;
        Self { n, n_mzis }
    }

    /// Decompose unitary matrix U into MZI phases.
    pub fn decompose(&self, u: &Array2<Complex64>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        assert_eq!(u.shape(), &[self.n, self.n]);

        let mut t = u.clone();
        let mut thetas = Vec::new();
        let mut phis = Vec::new();

        // Nullify off-diagonal elements column by column
        for k in 0..(self.n - 1) {
            if k % 2 == 0 {
                // Even column: nullify bottom-up
                for m in (k + 1..self.n).rev() {
                    let (theta, phi) = self.nullify_element(&t, m - 1, m, k);
                    thetas.push(theta);
                    phis.push(phi);
                    self.apply_rotation_left(&mut t, m - 1, m, theta, phi);
                }
            } else {
                // Odd column: nullify top-down
                for m in (k + 1)..self.n {
                    let (theta, phi) = self.nullify_element_right(&t, k, m - 1, m);
                    thetas.push(theta);
                    phis.push(phi);
                    self.apply_rotation_right(&mut t, m - 1, m, theta, phi);
                }
            }
        }

        // Remaining diagonal phases
        let output_phases: Vec<f64> = (0..self.n).map(|i| t[[i, i]].arg()).collect();

        (thetas, phis, output_phases)
    }

    fn nullify_element(&self, t: &Array2<Complex64>, i: usize, j: usize, k: usize) -> (f64, f64) {
        let a = t[[i, k]];
        let b = t[[j, k]];

        if b.norm() < 1e-12 {
            return (0.0, 0.0);
        }

        let theta = b.norm().atan2(a.norm());
        let phi = a.arg() - b.arg();

        (theta, phi)
    }

    fn nullify_element_right(
        &self,
        t: &Array2<Complex64>,
        k: usize,
        i: usize,
        j: usize,
    ) -> (f64, f64) {
        let a = t[[k, i]];
        let b = t[[k, j]];

        if b.norm() < 1e-12 {
            return (0.0, 0.0);
        }

        let theta = b.norm().atan2(a.norm());
        let phi = a.arg() - b.arg();

        (theta, phi)
    }

    fn apply_rotation_left(
        &self,
        t: &mut Array2<Complex64>,
        i: usize,
        j: usize,
        theta: f64,
        phi: f64,
    ) {
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let exp_phi_conj = Complex64::from_polar(1.0, -phi);

        for col in 0..self.n {
            let ti = t[[i, col]];
            let tj = t[[j, col]];

            t[[i, col]] = exp_phi_conj * cos_t * ti + exp_phi_conj * sin_t * tj;
            t[[j, col]] = -sin_t * ti + cos_t * tj;
        }
    }

    fn apply_rotation_right(
        &self,
        t: &mut Array2<Complex64>,
        i: usize,
        j: usize,
        theta: f64,
        phi: f64,
    ) {
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let exp_phi = Complex64::from_polar(1.0, phi);

        for row in 0..self.n {
            let ti = t[[row, i]];
            let tj = t[[row, j]];

            t[[row, i]] = cos_t * ti - sin_t * tj;
            t[[row, j]] = exp_phi * sin_t * ti + exp_phi * cos_t * tj;
        }
    }

    /// Reconstruct unitary matrix from MZI phases.
    pub fn reconstruct(
        &self,
        thetas: &[f64],
        phis: &[f64],
        output_phases: &[f64],
    ) -> Array2<Complex64> {
        let mut mesh = MZIMesh::new(self.n, 0.0, 0.0, -100.0);
        mesh.set_phases(thetas, phis, output_phases);
        mesh.get_matrix(false)
    }
}

/// Generate a random unitary matrix using QR decomposition.
pub fn random_unitary(n: usize) -> Array2<Complex64> {
    let mut rng = rand::thread_rng();

    // Random complex matrix
    let mut z = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            z[[i, j]] = Complex64::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ) / 2.0_f64.sqrt();
        }
    }

    // Simple Gram-Schmidt orthogonalization
    let mut q: Array2<Complex64> = Array2::zeros((n, n));

    for i in 0..n {
        let mut v = z.column(i).to_owned();

        // Subtract projections onto previous columns
        for j in 0..i {
            let q_j = q.column(j);
            let proj: Complex64 = q_j.iter().zip(v.iter()).map(|(a, b)| a.conj() * b).sum();
            for k in 0..n {
                v[k] -= proj * q[[k, j]];
            }
        }

        // Normalize
        let norm: f64 = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for k in 0..n {
                q[[k, i]] = v[k] / norm;
            }
        }
    }

    q
}

/// Compute reconstruction error for a decomposition.
pub fn decomposition_error(
    u: &Array2<Complex64>,
    thetas: &[f64],
    phis: &[f64],
    output_phases: &[f64],
) -> f64 {
    let n = u.shape()[0];
    let decomp = ClementsDecomposition::new(n);
    let u_reconstructed = decomp.reconstruct(thetas, phis, output_phases);

    let diff = u - &u_reconstructed;
    diff.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposition_small() {
        let u = random_unitary(4);
        let decomp = ClementsDecomposition::new(4);

        let (thetas, phis, output) = decomp.decompose(&u);

        assert_eq!(thetas.len(), 6); // 4*3/2
        assert_eq!(phis.len(), 6);
        assert_eq!(output.len(), 4);

        let error = decomposition_error(&u, &thetas, &phis, &output);
        assert!(error < 1e-6, "Error too large: {}", error);
    }

    #[test]
    fn test_decomposition_medium() {
        let u = random_unitary(8);
        let decomp = ClementsDecomposition::new(8);

        let (thetas, phis, output) = decomp.decompose(&u);
        let error = decomposition_error(&u, &thetas, &phis, &output);

        assert!(error < 1e-5, "Error too large: {}", error);
    }

    #[test]
    fn test_random_unitary_is_unitary() {
        let u = random_unitary(8);
        let u_h = u.t().mapv(|x| x.conj());
        let product = u.dot(&u_h);

        for i in 0..8 {
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (product[[i, j]].norm() - expected).abs();
                assert!(diff < 1e-6, "Not unitary at [{},{}]: {}", i, j, diff);
            }
        }
    }
}
