"""
Benchmark: PhotonCore-X Tinygrad Backend vs NumPy/CPU

Compares performance of optical backend for common NN operations.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from photonic_backend import (
    PhotonicDevice, PhotonicTensor, PhotonicMLP,
    tensor, randn
)


def benchmark_matmul():
    """Benchmark matrix multiplication throughput."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Matrix Multiplication")
    print("=" * 60)

    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]

    for m, n in sizes:
        k = m  # Square matrices

        # NumPy baseline
        a_np = np.random.randn(m, k).astype(np.float32)
        b_np = np.random.randn(k, n).astype(np.float32)

        # Warmup
        for _ in range(5):
            _ = a_np @ b_np

        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = a_np @ b_np
        numpy_time = (time.perf_counter() - start) / n_iters

        # Photonic backend
        device = PhotonicDevice(n_ports=min(64, m), n_wdm_channels=16)
        a_ph = tensor(a_np, device)
        b_ph = tensor(b_np, device)

        # Warmup
        for _ in range(5):
            _ = a_ph @ b_ph

        start = time.perf_counter()
        for _ in range(n_iters):
            _ = a_ph @ b_ph
        photonic_time = (time.perf_counter() - start) / n_iters

        ops = 2 * m * k * n
        numpy_gops = ops / numpy_time / 1e9
        photonic_gops = ops / photonic_time / 1e9

        print(f"\n  {m}x{k} @ {k}x{n}:")
        print(f"    NumPy:    {numpy_time*1e6:8.1f} μs  ({numpy_gops:.2f} GOps/s)")
        print(f"    Photonic: {photonic_time*1e6:8.1f} μs  ({photonic_gops:.2f} GOps/s)")

        # Note: In simulation, photonic is slower due to Python overhead
        # Real hardware would be faster due to O(1) optical matrix multiply


def benchmark_mlp():
    """Benchmark MLP inference."""
    print("\n" + "=" * 60)
    print("BENCHMARK: MLP Inference")
    print("=" * 60)

    configs = [
        ([784, 256, 10], "Small (MNIST)"),
        ([784, 512, 256, 10], "Medium"),
        ([784, 1024, 512, 256, 10], "Large"),
    ]

    batch_size = 32

    for layer_sizes, name in configs:
        # Photonic MLP
        device = PhotonicDevice(n_ports=64, n_wdm_channels=32)
        mlp = PhotonicMLP(layer_sizes, device, activation='relu')

        x = tensor(np.random.randn(batch_size, layer_sizes[0]), device)

        # Warmup
        for _ in range(5):
            _ = mlp(x)

        n_iters = 50
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = mlp(x)
        elapsed = (time.perf_counter() - start) / n_iters

        # Calculate total ops
        total_ops = 0
        for i in range(len(layer_sizes) - 1):
            total_ops += 2 * batch_size * layer_sizes[i] * layer_sizes[i+1]

        gops = total_ops / elapsed / 1e9

        print(f"\n  {name}: {layer_sizes}")
        print(f"    Batch: {batch_size}")
        print(f"    Time:  {elapsed*1e3:.2f} ms")
        print(f"    Throughput: {gops:.2f} GOps/s")
        print(f"    Latency/sample: {elapsed/batch_size*1e3:.2f} ms")


def benchmark_activations():
    """Benchmark optical nonlinearities."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Optical Activations")
    print("=" * 60)

    size = 10000
    device = PhotonicDevice(n_ports=64)

    x = tensor(np.random.randn(size), device)

    activations = ['relu', 'sigmoid', 'tanh', 'softmax']

    for act_name in activations:
        # Warmup
        for _ in range(10):
            if act_name == 'relu':
                _ = x.relu()
            elif act_name == 'sigmoid':
                _ = x.sigmoid()
            elif act_name == 'tanh':
                _ = x.tanh()
            elif act_name == 'softmax':
                _ = x.softmax()

        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            if act_name == 'relu':
                _ = x.relu()
            elif act_name == 'sigmoid':
                _ = x.sigmoid()
            elif act_name == 'tanh':
                _ = x.tanh()
            elif act_name == 'softmax':
                _ = x.softmax()
        elapsed = (time.perf_counter() - start) / n_iters

        throughput = size / elapsed / 1e6

        print(f"  {act_name:10s}: {elapsed*1e6:8.1f} μs  ({throughput:.1f} M elements/s)")


def benchmark_wdm_parallelism():
    """Benchmark WDM channel utilization."""
    print("\n" + "=" * 60)
    print("BENCHMARK: WDM Parallelism")
    print("=" * 60)

    m, k, n = 256, 256, 256
    a_np = np.random.randn(m, k).astype(np.float32)
    b_np = np.random.randn(k, n).astype(np.float32)

    wdm_channels = [1, 4, 16, 64, 128]

    for n_wdm in wdm_channels:
        device = PhotonicDevice(n_ports=64, n_wdm_channels=n_wdm)
        a = tensor(a_np, device)
        b = tensor(b_np, device)

        # Warmup
        for _ in range(3):
            _ = a @ b

        n_iters = 20
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = a @ b
        elapsed = (time.perf_counter() - start) / n_iters

        ops = 2 * m * k * n
        gops = ops / elapsed / 1e9

        print(f"  {n_wdm:3d} WDM channels: {elapsed*1e3:6.2f} ms  ({gops:.2f} GOps/s)")


def benchmark_energy_model():
    """Estimate energy efficiency."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Energy Efficiency Model")
    print("=" * 60)

    # PhotonCore-X energy model (estimated from design spec)
    # Real chip: ~50W total power
    # Simulation: estimate based on ops

    device = PhotonicDevice(n_ports=64, n_wdm_channels=128)

    # Run workload
    mlp = PhotonicMLP([784, 512, 256, 10], device)
    x = tensor(np.random.randn(1000, 784), device)

    start = time.perf_counter()
    _ = mlp(x)
    elapsed = time.perf_counter() - start

    stats = device.stats()

    # Energy model
    chip_power_w = 50.0  # Target chip power
    ops = stats['total_ops']
    ops_per_joule = ops / (chip_power_w * elapsed)

    print(f"  Total operations: {ops:,}")
    print(f"  Time: {elapsed*1e3:.2f} ms")
    print(f"  Chip power: {chip_power_w} W")
    print(f"  Energy efficiency: {ops_per_joule:.2e} OPs/J")

    # Compare to GPU
    gpu_efficiency = 1e12  # ~1 TOPs/J for modern GPU
    improvement = ops_per_joule / gpu_efficiency

    print(f"\n  vs GPU (~1 TOPs/J): {improvement:.1f}x")
    print(f"  (Note: Simulation - real hardware would be faster)")


def main():
    print("\n" + "#" * 60)
    print("#  PhotonCore-X Tinygrad Backend Benchmark")
    print("#" * 60)

    benchmark_matmul()
    benchmark_mlp()
    benchmark_activations()
    benchmark_wdm_parallelism()
    benchmark_energy_model()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
