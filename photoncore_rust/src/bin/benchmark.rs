//! PhotonCore-X Benchmark Suite

use photoncore::clements::{random_unitary, ClementsDecomposition};
use photoncore::mzi::{create_random_unitary_mesh, MZIMesh};
use photoncore::optical_mvm::OpticalMatrixUnit;

use ndarray::Array1;
use num_complex::Complex64;
use std::time::Instant;

fn benchmark_decomposition() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: Matrix Decomposition");
    println!("{}", "=".repeat(60));

    let sizes = [8, 16, 32, 64, 128];

    for &n in &sizes {
        let decomp = ClementsDecomposition::new(n);
        let u = random_unitary(n);

        let n_iters = if n <= 32 { 100 } else { 10 };

        let start = Instant::now();
        for _ in 0..n_iters {
            let _ = decomp.decompose(&u);
        }
        let elapsed = start.elapsed().as_secs_f64() / n_iters as f64;

        let n_mzis = n * (n - 1) / 2;
        println!("  {}x{}: {:.3} ms ({} MZIs)", n, n, elapsed * 1000.0, n_mzis);
    }
}

fn benchmark_forward_pass() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: Forward Pass (Matrix-Vector Multiply)");
    println!("{}", "=".repeat(60));

    let sizes = [16, 32, 64, 128];

    for &n in &sizes {
        let mut unit = OpticalMatrixUnit::new(n, 0.1, 0.01, 0.001, 8);
        let u = random_unitary(n);
        unit.load_matrix(&u);

        let mut x = Array1::zeros(n);
        x[0] = Complex64::new(1.0, 0.0);

        // Warm up
        for _ in 0..10 {
            let _ = unit.forward(&x, false);
        }

        let n_iters = 1000;

        // Without noise
        let start = Instant::now();
        for _ in 0..n_iters {
            let _ = unit.forward(&x, false);
        }
        let elapsed_no_noise = start.elapsed().as_secs_f64() / n_iters as f64;

        // With noise
        let start = Instant::now();
        for _ in 0..n_iters {
            let _ = unit.forward(&x, true);
        }
        let elapsed_noise = start.elapsed().as_secs_f64() / n_iters as f64;

        let ops = n * n;
        let throughput = ops as f64 / elapsed_no_noise / 1e9;

        println!(
            "  {}x{}: {:.1} μs (no noise), {:.1} μs (noise)",
            n,
            n,
            elapsed_no_noise * 1e6,
            elapsed_noise * 1e6
        );
        println!("         Throughput: {:.2} GOps/s", throughput);
    }
}

fn benchmark_mesh_forward() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: Raw MZI Mesh Forward Pass");
    println!("{}", "=".repeat(60));

    let sizes = [8, 16, 32, 64, 128, 256];

    for &n in &sizes {
        let mesh = create_random_unitary_mesh(n);

        let mut x = Array1::zeros(n);
        x[0] = Complex64::new(1.0, 0.0);

        // Warm up
        for _ in 0..10 {
            let _ = mesh.forward(&x, false);
        }

        let n_iters = if n <= 128 { 1000 } else { 100 };

        let start = Instant::now();
        for _ in 0..n_iters {
            let _ = mesh.forward(&x, false);
        }
        let elapsed = start.elapsed().as_secs_f64() / n_iters as f64;

        let ops = n * n;
        let throughput = ops as f64 / elapsed / 1e9;

        println!(
            "  {}x{}: {:.1} μs, {:.2} GOps/s",
            n,
            n,
            elapsed * 1e6,
            throughput
        );
    }
}

fn benchmark_energy_efficiency() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: Energy Efficiency (Estimated)");
    println!("{}", "=".repeat(60));

    let chip_power_w = 50.0;
    let sizes = [64, 128, 256];

    for &n in &sizes {
        let mesh = create_random_unitary_mesh(n);

        let mut x = Array1::zeros(n);
        x[0] = Complex64::new(1.0, 0.0);

        let n_iters = 1000;
        let start = Instant::now();
        for _ in 0..n_iters {
            let _ = mesh.forward(&x, false);
        }
        let elapsed = start.elapsed().as_secs_f64();

        let ops_per_iter = n * n;
        let ops_per_sec = (n_iters * ops_per_iter) as f64 / elapsed;
        let ops_per_joule = ops_per_sec / chip_power_w;

        println!("  {}x{}: {:.2e} OPs/J", n, n, ops_per_joule);
    }
}

fn main() {
    println!("\n{}", "#".repeat(60));
    println!("#  PhotonCore-X Rust Benchmark Suite");
    println!("{}", "#".repeat(60));

    benchmark_decomposition();
    benchmark_forward_pass();
    benchmark_mesh_forward();
    benchmark_energy_efficiency();

    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK COMPLETE");
    println!("{}", "=".repeat(60));
}
