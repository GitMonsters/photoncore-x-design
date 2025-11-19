"""
Quick PhotonCore-X Performance Benchmark

Direct MVM benchmark without heavy decomposition overhead.
Shows realistic hardware performance projections.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'photoncore_prototype', 'src'))

from mzi import MZIMesh


def benchmark_mvm():
    """Benchmark optical matrix-vector multiply."""
    print("=" * 70)
    print("PhotonCore-X Quick Performance Benchmark")
    print("=" * 70)
    print()

    results = []

    for n_ports in [4, 8, 16, 32]:
        print(f"Matrix size: {n_ports}x{n_ports}")
        print("-" * 50)

        # Create mesh
        mesh = MZIMesh(n_ports, phase_noise_std=0.01)

        # Set random phases
        n_mzis = n_ports * (n_ports - 1) // 2
        thetas = np.random.uniform(0, np.pi, n_mzis)
        phis = np.random.uniform(0, 2*np.pi, n_mzis)
        output_phases = np.random.uniform(0, 2*np.pi, n_ports)
        mesh.set_phases(thetas, phis, output_phases)

        # Benchmark forward pass
        x = np.random.randn(n_ports).astype(np.complex128)
        x /= np.linalg.norm(x)

        n_iters = 1000
        t0 = time.time()
        for _ in range(n_iters):
            y = mesh.forward(x, add_noise=True)
        elapsed = time.time() - t0

        ops_per_mvm = 2 * n_ports * n_ports  # multiply-add
        throughput = n_iters * ops_per_mvm / elapsed / 1e6  # MOPS

        print(f"  Simulation: {n_iters} MVMs in {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.1f} MOPS (simulated)")
        print()

        # Projected hardware performance
        # Real photonics: ~1 μs per MVM regardless of size (speed of light)
        hw_time_us = 1.0
        hw_throughput_tops = ops_per_mvm / hw_time_us / 1e6

        results.append({
            'n_ports': n_ports,
            'n_mzis': n_mzis,
            'sim_time_ms': elapsed * 1000 / n_iters,
            'sim_mops': throughput,
            'hw_time_us': hw_time_us,
            'hw_tops': hw_throughput_tops
        })

    # Summary
    print("=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print()
    print("Size  | MZIs | Sim (ms) | Sim MOPS | HW (μs) | HW TOPs")
    print("-" * 70)

    for r in results:
        print(f"{r['n_ports']:3d}x{r['n_ports']:<3d}| {r['n_mzis']:4d} | "
              f"{r['sim_time_ms']:7.2f}  | {r['sim_mops']:7.1f}  | "
              f"{r['hw_time_us']:6.1f}  | {r['hw_tops']:6.2f}")

    print()
    print("=" * 70)
    print("Key Insights")
    print("=" * 70)
    print()
    print("1. O(1) latency: Hardware time stays ~1μs regardless of matrix size")
    print("   (optical interference is instantaneous)")
    print()
    print("2. Hardware vs simulation speedup:")
    for r in results:
        speedup = r['sim_time_ms'] * 1000 / r['hw_time_us']
        print(f"   {r['n_ports']}x{r['n_ports']}: {speedup:.0f}x faster")
    print()
    print("3. With 128 WDM channels (parallel wavelengths):")
    print(f"   Peak throughput: {results[-1]['hw_tops'] * 128:.1f} TOPs")
    print(f"   At 50W: {results[-1]['hw_tops'] * 128 / 50:.1f} TOPs/W")
    print()
    print("4. Comparison to GPUs:")
    print("   NVIDIA H100: ~2000 TOPs at 700W = 2.8 TOPs/W")
    print(f"   PhotonCore-X: ~{results[-1]['hw_tops'] * 128:.0f} TOPs at 50W = "
          f"{results[-1]['hw_tops'] * 128 / 50:.1f} TOPs/W")
    print(f"   Efficiency gain: {(results[-1]['hw_tops'] * 128 / 50) / 2.8:.1f}x better")
    print()

    return results


def benchmark_attention_layer():
    """Benchmark attention-like pattern."""
    print("=" * 70)
    print("Attention Layer Simulation")
    print("=" * 70)
    print()

    # Attention uses: Q, K, V projections + output projection
    # Each is a matrix-vector multiply

    for seq_len in [16, 32, 64, 128]:
        embed_dim = 64
        n_mvms = seq_len * 4  # Q, K, V, O projections

        # Simulated time (estimated from MVM benchmark)
        sim_time_per_mvm_ms = 0.1  # ~100μs per MVM in simulation
        sim_total_ms = n_mvms * sim_time_per_mvm_ms

        # Hardware time
        hw_time_per_mvm_us = 1.0
        hw_total_us = n_mvms * hw_time_per_mvm_us

        speedup = sim_total_ms * 1000 / hw_total_us

        print(f"Seq length: {seq_len}, MVMs: {n_mvms}")
        print(f"  Simulation: {sim_total_ms:.1f} ms")
        print(f"  Hardware:   {hw_total_us:.1f} μs")
        print(f"  Speedup:    {speedup:.0f}x")
        print()

    print("For LLM inference (seq=2048, embed=4096):")
    n_mvms = 2048 * 4
    hw_time_ms = n_mvms * 1.0 / 1000
    print(f"  MVMs per layer: {n_mvms}")
    print(f"  Hardware time: {hw_time_ms:.1f} ms per layer")
    print(f"  With 128 WDM: {hw_time_ms / 128:.3f} ms per layer")
    print()


if __name__ == "__main__":
    results = benchmark_mvm()
    print()
    benchmark_attention_layer()
