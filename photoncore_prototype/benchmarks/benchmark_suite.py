#!/usr/bin/env python3
"""
PhotonCore-X Benchmark Suite

Comprehensive benchmarks for photonic AI accelerator performance.
"""

import numpy as np
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mzi import MZIMesh
from clements import ClementsDecomposition, random_unitary
from optical_mvm import OpticalMatrixUnit, WDMParallelUnit
from photoncore import PhotonCoreSimulator, create_optical_mlp


def benchmark_matrix_decomposition():
    """Benchmark Clements decomposition speed."""
    print("\n" + "="*60)
    print("BENCHMARK: Matrix Decomposition")
    print("="*60)

    sizes = [8, 16, 32, 64, 128]
    results = []

    for n in sizes:
        decomp = ClementsDecomposition(n)

        # Generate random unitary
        U = random_unitary(n)

        # Time decomposition
        start = time.time()
        n_iters = 100 if n <= 32 else 10
        for _ in range(n_iters):
            thetas, phis, output = decomp.decompose(U)
        elapsed = (time.time() - start) / n_iters

        results.append({
            'size': n,
            'n_mzis': n*(n-1)//2,
            'time_ms': elapsed * 1000
        })

        print(f"  {n}x{n}: {elapsed*1000:.3f} ms ({n*(n-1)//2} MZIs)")

    return results


def benchmark_forward_pass():
    """Benchmark optical matrix-vector multiplication."""
    print("\n" + "="*60)
    print("BENCHMARK: Forward Pass (Matrix-Vector Multiply)")
    print("="*60)

    sizes = [16, 32, 64, 128]
    results = []

    for n in sizes:
        unit = OpticalMatrixUnit(n)
        U = random_unitary(n)
        unit.load_matrix(U)

        x = np.random.randn(n) + 1j * np.random.randn(n)
        x /= np.linalg.norm(x)

        # Warm up
        for _ in range(10):
            unit.forward(x)

        # Benchmark
        n_iters = 1000
        start = time.time()
        for _ in range(n_iters):
            y = unit.forward(x, add_noise=False)
        elapsed = (time.time() - start) / n_iters

        # With noise
        start_noise = time.time()
        for _ in range(n_iters):
            y = unit.forward(x, add_noise=True)
        elapsed_noise = (time.time() - start_noise) / n_iters

        ops = n * n  # n^2 multiplications
        throughput = ops / elapsed

        results.append({
            'size': n,
            'time_no_noise_us': elapsed * 1e6,
            'time_with_noise_us': elapsed_noise * 1e6,
            'throughput_gops': throughput / 1e9
        })

        print(f"  {n}x{n}: {elapsed*1e6:.1f} μs (no noise), {elapsed_noise*1e6:.1f} μs (noise)")
        print(f"         Throughput: {throughput/1e9:.2f} GOps/s")

    return results


def benchmark_accuracy():
    """Benchmark computational accuracy with noise."""
    print("\n" + "="*60)
    print("BENCHMARK: Accuracy vs Noise")
    print("="*60)

    n = 32
    noise_levels = [0.0, 0.001, 0.01, 0.05, 0.1]
    results = []

    for noise_std in noise_levels:
        unit = OpticalMatrixUnit(
            n,
            phase_noise_std=noise_std,
            detector_noise_std=noise_std/10
        )

        U = random_unitary(n)
        unit.load_matrix(U)

        # Test many vectors
        errors = []
        for _ in range(100):
            x = np.random.randn(n)
            x /= np.linalg.norm(x)

            expected = U @ x
            actual = unit.forward(x, add_noise=True)

            error = np.linalg.norm(expected - actual) / np.linalg.norm(expected)
            errors.append(error)

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        # Estimate effective bits
        if mean_error > 0:
            effective_bits = -np.log2(mean_error)
        else:
            effective_bits = 16

        results.append({
            'noise_std': noise_std,
            'mean_error': mean_error,
            'max_error': max_error,
            'effective_bits': effective_bits
        })

        print(f"  Noise σ={noise_std:.3f}: Mean error={mean_error:.2e}, "
              f"Effective bits={effective_bits:.1f}")

    return results


def benchmark_wdm_parallel():
    """Benchmark WDM parallel processing."""
    print("\n" + "="*60)
    print("BENCHMARK: WDM Parallel Processing")
    print("="*60)

    n = 32
    channel_counts = [4, 16, 64, 128]
    results = []

    for n_channels in channel_counts:
        wdm = WDMParallelUnit(n, n_channels)

        # Load random matrices
        matrices = [random_unitary(n) for _ in range(n_channels)]
        wdm.load_matrices(matrices)

        # Create inputs
        inputs = [np.random.randn(n) for _ in range(n_channels)]

        # Warm up
        for _ in range(5):
            wdm.forward_parallel(inputs)

        # Benchmark
        n_iters = 100
        start = time.time()
        for _ in range(n_iters):
            outputs = wdm.forward_parallel(inputs, add_noise=False)
        elapsed = (time.time() - start) / n_iters

        ops = n_channels * n * n
        throughput = ops / elapsed

        results.append({
            'n_channels': n_channels,
            'time_ms': elapsed * 1000,
            'throughput_gops': throughput / 1e9,
            'speedup_vs_serial': n_channels  # Ideal speedup
        })

        print(f"  {n_channels} channels: {elapsed*1000:.2f} ms, "
              f"Throughput: {throughput/1e9:.1f} GOps/s")

    return results


def benchmark_neural_network():
    """Benchmark neural network inference."""
    print("\n" + "="*60)
    print("BENCHMARK: Neural Network Inference")
    print("="*60)

    architectures = [
        [64, 32, 10],
        [128, 64, 32, 10],
        [256, 128, 64, 10],
        [784, 256, 128, 10]  # MNIST-like
    ]

    results = []

    for arch in architectures:
        network = create_optical_mlp(arch)
        network.eval()

        x = np.random.randn(arch[0])

        # Warm up
        for _ in range(10):
            network(x)

        # Benchmark
        n_iters = 500
        start = time.time()
        for _ in range(n_iters):
            y = network(x)
        elapsed = (time.time() - start) / n_iters

        latency_ms = elapsed * 1000
        throughput = 1 / elapsed

        # Count operations
        ops = sum(arch[i] * arch[i+1] for i in range(len(arch)-1))

        results.append({
            'architecture': arch,
            'latency_ms': latency_ms,
            'throughput_samples_per_s': throughput,
            'total_ops': ops
        })

        arch_str = '→'.join(map(str, arch))
        print(f"  {arch_str}: {latency_ms:.2f} ms, {throughput:.0f} samples/s")

    return results


def benchmark_energy_efficiency():
    """Estimate energy efficiency."""
    print("\n" + "="*60)
    print("BENCHMARK: Energy Efficiency (Estimated)")
    print("="*60)

    # Assumptions
    chip_power_w = 50  # PhotonCore-X power consumption
    gpu_power_w = 700  # H100 power

    # Matrix sizes
    sizes = [64, 128, 256, 512]

    print("\nComparing PhotonCore-X vs NVIDIA H100:\n")

    for n in sizes:
        # PhotonCore throughput (simulated)
        sim = PhotonCoreSimulator(n_ports=n)
        U = random_unitary(n)
        sim.load_matrix(U)

        x = np.random.randn(n)

        # Benchmark
        n_iters = 1000
        start = time.time()
        for _ in range(n_iters):
            sim.matmul(x, add_noise=False)
        elapsed = time.time() - start

        ops_per_iter = n * n
        photoncore_ops_per_s = (n_iters * ops_per_iter) / elapsed
        photoncore_ops_per_j = photoncore_ops_per_s / chip_power_w

        # H100 estimate (peak ~2000 TFLOPS for fp16)
        # Real efficiency is lower for small matrices
        h100_ops_per_s = 2e15 * (n / 1024) ** 2  # Scale down for small matrices
        h100_ops_per_j = h100_ops_per_s / gpu_power_w

        efficiency_ratio = photoncore_ops_per_j / h100_ops_per_j

        print(f"  {n}x{n} matrix:")
        print(f"    PhotonCore-X: {photoncore_ops_per_j:.2e} OPs/J")
        print(f"    H100 (est):   {h100_ops_per_j:.2e} OPs/J")
        print(f"    Ratio:        {efficiency_ratio:.1f}×")
        print()


def run_all_benchmarks():
    """Run complete benchmark suite."""
    print("\n" + "#"*60)
    print("#  PhotonCore-X Benchmark Suite")
    print("#"*60)

    results = {}

    results['decomposition'] = benchmark_matrix_decomposition()
    results['forward_pass'] = benchmark_forward_pass()
    results['accuracy'] = benchmark_accuracy()
    results['wdm_parallel'] = benchmark_wdm_parallel()
    results['neural_network'] = benchmark_neural_network()
    benchmark_energy_efficiency()

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)

    return results


if __name__ == "__main__":
    run_all_benchmarks()
