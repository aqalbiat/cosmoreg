import numpy as np
from augment.augmentation import add_gaussian_noise

def test_shape_and_type():
    # Create a dummy batch of 2 maps, 256x256
    dummy_maps = np.ones((2, 256, 256), dtype=np.float32)
    augmented = add_gaussian_noise(dummy_maps, alpha=0.05)
    
    assert augmented.shape == dummy_maps.shape
    assert augmented.dtype == np.float32

def test_no_side_effects():
    # Ensure the original array is not modified in-place
    dummy_maps = np.ones((2, 256, 256), dtype=np.float32)
    dummy_maps_copy = dummy_maps.copy()
    
    _ = add_gaussian_noise(dummy_maps, alpha=0.05)
    
    np.testing.assert_array_equal(dummy_maps, dummy_maps_copy)

def test_physical_constraint():
    # Create maps with values exactly at 0
    dummy_maps = np.zeros((2, 256, 256), dtype=np.float32)
    # Add huge noise to force negative values
    augmented = add_gaussian_noise(dummy_maps, alpha=10.0)
    
    # Assert no values are below 0
    assert np.min(augmented) >= 0.0

def test_reproducibility():
    # Ensure the same seed produces the same noise
    rng = np.random.default_rng(99)
    dummy_maps = rng.random((2, 256, 256)).astype(np.float32)
    
    aug1 = add_gaussian_noise(dummy_maps, alpha=0.1, seed=42)
    aug2 = add_gaussian_noise(dummy_maps, alpha=0.1, seed=42)
    
    np.testing.assert_array_equal(aug1, aug2)