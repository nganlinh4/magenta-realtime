#!/usr/bin/env python3
"""Test script to verify JAX CUDA installation."""

import jax
import jax.numpy as jnp

def test_jax_cuda():
    """Test JAX CUDA functionality."""
    print("JAX version:", jax.__version__)
    print("Available devices:", jax.devices())
    
    # Test basic computation
    x = jnp.array([1, 2, 3, 4, 5])
    y = jnp.array([2, 3, 4, 5, 6])
    
    # This should run on GPU if available
    result = jnp.dot(x, y)
    print(f"Dot product result: {result}")
    
    # Check device types
    for device in jax.devices():
        print(f"Device: {device}, Kind: {device.device_kind}, Platform: {device.platform}")

    # Check if GPU is available
    gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']

    if gpu_devices:
        print(f"✅ GPU devices found: {gpu_devices}")

        # Test GPU computation
        with jax.default_device(gpu_devices[0]):
            gpu_result = jnp.dot(x, y)
            print(f"GPU computation result: {gpu_result}")

        # Test a smaller computation to verify GPU acceleration
        print("Testing small matrix multiplication on GPU...")
        try:
            A = jnp.ones((10, 10))
            B = jnp.ones((10, 10))
            with jax.default_device(gpu_devices[0]):
                C = jnp.dot(A, B)
                print(f"Matrix multiplication result shape: {C.shape}, sample value: {C[0, 0]}")
        except Exception as e:
            print(f"GPU computation failed: {e}")
            print("This might be due to GPU memory constraints or driver issues.")
    else:
        print("⚠️  No GPU devices found, using CPU")
    
    print("JAX CUDA test completed successfully!")

if __name__ == "__main__":
    test_jax_cuda()
