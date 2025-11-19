#!/usr/bin/env python3
"""
MNIST Inference on PhotonCore-X

Demonstrates optical neural network inference on handwritten digits.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from photoncore import PhotonCoreSimulator, PhotonCoreSDK, create_optical_mlp


def load_mnist_sample():
    """
    Load sample MNIST data.

    Returns simulated data if sklearn not available.
    """
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        print("Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)

        # Normalize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1000, random_state=42
        )

        return X_test[:100], y_test[:100]

    except ImportError:
        print("sklearn not available, using simulated MNIST data")

        # Create fake MNIST-like data
        n_samples = 100
        X = np.random.randn(n_samples, 784)
        y = np.random.randint(0, 10, n_samples)

        return X, y


def create_pretrained_weights(input_dim, hidden_dims, output_dim):
    """
    Create simulated pre-trained weights.

    In practice, these would come from training on actual MNIST.
    """
    layers = [input_dim] + hidden_dims + [output_dim]
    weights = []
    biases = []

    for i in range(len(layers) - 1):
        # Xavier initialization
        std = np.sqrt(2.0 / (layers[i] + layers[i+1]))
        W = np.random.randn(layers[i+1], layers[i]) * std
        b = np.zeros(layers[i+1])

        weights.append(W)
        biases.append(b)

    return weights, biases


def run_inference_demo():
    """Run MNIST inference demonstration."""
    print("\n" + "="*60)
    print("PhotonCore-X MNIST Inference Demo")
    print("="*60)

    # Load data
    X_test, y_test = load_mnist_sample()
    print(f"\nLoaded {len(X_test)} test samples")

    # Create optical neural network
    # Architecture: 784 -> 256 -> 128 -> 10
    print("\nCreating optical neural network...")
    network = create_optical_mlp([784, 256, 128, 10], activation='relu')
    network.eval()

    # Get pretrained weights (simulated)
    weights, biases = create_pretrained_weights(784, [256, 128], 10)

    # Load weights into optical network
    for i, (layer_name, layer) in enumerate(network.layers):
        if layer_name == 'linear':
            # Find corresponding weight matrix
            w_idx = sum(1 for n, _ in network.layers[:i] if n == 'linear')
            if w_idx < len(weights):
                layer.set_weights(weights[w_idx])
                layer.bias = biases[w_idx]

    # Run inference
    print("\nRunning inference...")

    predictions = []
    confidences = []

    for i, x in enumerate(X_test):
        # Forward pass
        output = network(x)

        # Get prediction (optical softmax approximation)
        output_real = np.real(output)
        pred = np.argmax(output_real)
        conf = np.max(output_real) / (np.sum(np.abs(output_real)) + 1e-10)

        predictions.append(pred)
        confidences.append(conf)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(X_test)} samples")

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    # Calculate accuracy
    # Note: With random weights, accuracy will be ~10% (random chance)
    accuracy = np.mean(predictions == y_test)

    print(f"\n{'='*60}")
    print("Results:")
    print(f"  Samples: {len(X_test)}")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    print(f"{'='*60}")

    # Show some predictions
    print("\nSample predictions:")
    for i in range(10):
        print(f"  Sample {i}: Predicted={predictions[i]}, "
              f"Actual={y_test[i]}, "
              f"Confidence={confidences[i]:.3f}")

    return accuracy


def compare_noise_levels():
    """Compare accuracy under different noise conditions."""
    print("\n" + "="*60)
    print("Noise Analysis")
    print("="*60)

    X_test, y_test = load_mnist_sample()

    noise_levels = [0.0, 0.01, 0.05, 0.1]
    results = []

    for noise in noise_levels:
        # Create network with specific noise level
        from optical_mvm import OpticalNeuralLayer

        layers = [784, 256, 128, 10]
        network_layers = []

        for i in range(len(layers) - 1):
            layer = OpticalNeuralLayer(
                layers[i], layers[i+1],
                'relu' if i < len(layers) - 2 else 'none'
            )
            layer.matrix_unit.phase_noise_std = noise
            layer.matrix_unit.detector_noise_std = noise / 10
            network_layers.append(layer)

        # Run inference
        predictions = []
        for x in X_test:
            out = x
            for layer in network_layers:
                out = layer.forward(out)
            predictions.append(np.argmax(np.real(out)))

        accuracy = np.mean(np.array(predictions) == y_test)
        results.append((noise, accuracy))

        print(f"  Noise σ={noise:.2f}: Accuracy={accuracy*100:.1f}%")

    return results


def benchmark_inference_speed():
    """Benchmark inference speed."""
    print("\n" + "="*60)
    print("Inference Speed Benchmark")
    print("="*60)

    import time

    # Create network
    network = create_optical_mlp([784, 256, 128, 10])
    network.eval()

    # Create test input
    x = np.random.randn(784)

    # Warm up
    for _ in range(10):
        network(x)

    # Benchmark
    n_iters = 1000
    start = time.time()
    for _ in range(n_iters):
        y = network(x)
    elapsed = time.time() - start

    latency_ms = (elapsed / n_iters) * 1000
    throughput = n_iters / elapsed

    print(f"\n  Latency: {latency_ms:.3f} ms/sample")
    print(f"  Throughput: {throughput:.0f} samples/second")

    # Compare with "GPU" (numpy)
    W1 = np.random.randn(256, 784)
    W2 = np.random.randn(128, 256)
    W3 = np.random.randn(10, 128)

    start = time.time()
    for _ in range(n_iters):
        h1 = np.maximum(0, W1 @ x)  # ReLU
        h2 = np.maximum(0, W2 @ h1)
        out = W3 @ h2
    elapsed_numpy = time.time() - start

    numpy_latency = (elapsed_numpy / n_iters) * 1000

    print(f"\n  NumPy comparison: {numpy_latency:.3f} ms/sample")
    print(f"  Ratio: {numpy_latency/latency_ms:.2f}×")

    return latency_ms, throughput


if __name__ == "__main__":
    # Run demo
    run_inference_demo()

    # Analyze noise effects
    compare_noise_levels()

    # Benchmark speed
    benchmark_inference_speed()

    print("\n✓ MNIST demo complete!")
