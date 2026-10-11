import numpy as np

def add_gaussian_noise(maps: np.ndarray, alpha: float = 0.05, seed: int = 42) -> np.ndarray:
    """
    Adds i.i.d. Gaussian noise to a batch of 2D density maps.
    
    The noise standard deviation is scaled relative to each map's own 
    spatial standard deviation.
    
    Parameters
    ----------
    maps : np.ndarray
        Original maps with shape [M, 256, 256].
    alpha : float
        Noise strength. 0.05 means the noise standard deviation is 5% 
        of the map's standard deviation.
    seed : int
        Random seed for strictly reproducible i.i.d. noise.
        
    Returns
    -------
    np.ndarray
        Augmented maps with the same shape, clipped to >= 0.
    """
    if maps.ndim != 3:
        raise ValueError(f"Expected maps with shape [M, H, W], got {maps.shape}.")

    if alpha < 0:
        raise ValueError("Alpha (noise strength) must be non-negative.")

    # Convert to float32 to match pipeline defaults and avoid modifying the original array
    maps_clean = np.array(maps, dtype=np.float32)

    # Calculate standard deviation per map. Shape becomes [M, 1, 1]
    # Calculated in float64 for precision, cast back to float32
    map_std = np.std(
        maps_clean, 
        axis=(1, 2), 
        keepdims=True, 
        dtype=np.float64
    ).astype(np.float32)

    # Generate i.i.d. Standard Normal noise (mean 0, std 1)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=maps_clean.shape, dtype=np.float32)

    # Scale noise by alpha and the map's own standard deviation
    scaled_noise = noise * (alpha * map_std)

    # Add noise to the maps
    augmented_maps = maps_clean + scaled_noise

    # PHYSICAL CONSTRAINT:
    # Matter density fields cannot have negative density.
    # Clip any values that fall below zero due to the Gaussian noise.
    augmented_maps = np.clip(augmented_maps, a_min=0.0, a_max=None)

    return augmented_maps