import argparse
import shutil
import numpy as np
from pathlib import Path

# Import your function
from augment.augmentation import add_gaussian_noise

def process_dataset(input_dir: str, output_dir: str, alpha: float, seed: int):
    in_path = Path(input_dir)
    if not in_path.exists():
        raise FileNotFoundError(f"Input directory {in_path} does not exist. Run the pipeline prepare step first.")

    out_root = Path(output_dir) / f"gaussian_alpha_{alpha}".replace(".", "")
    
    # Find all maps.npy files in the maps2d directory
    map_files = sorted(list(in_path.rglob("maps.npy")))
    if not map_files:
        print(f"No maps.npy found in {in_path}. Check your paths.")
        return

    print(f"Found {len(map_files)} map files. Starting augmentation (alpha={alpha})...")

    for map_file in map_files:
        # Determine the relative path (e.g., level1/fold1)
        rel_dir = map_file.parent.relative_to(in_path)
        out_dir = out_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing: {rel_dir}")

        # 1. Load, augment, and save the maps
        original_maps = np.load(map_file)
        augmented_maps = add_gaussian_noise(original_maps, alpha=alpha, seed=seed)
        np.save(out_dir / "maps.npy", augmented_maps)

        # 2. Copy all necessary metadata files exactly as they are
        for meta_file in ["labels.npy", "simulation_ids.npy", "map_ids.npy", "manifest.json"]:
            src_file = map_file.parent / meta_file
            if src_file.exists():
                shutil.copy2(src_file, out_dir / meta_file)

    print(f"✅ Augmentation complete! Data saved to: {out_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented CAMELS datasets.")
    parser.add_argument("--input", type=str, default="maps2d", help="Path to original maps2d folder")
    parser.add_argument("--output", type=str, default="augmented_maps2d", help="Path to save augmented data")
    parser.add_argument("--alpha", type=float, default=0.05, help="Noise strength (e.g., 0.05 for 5%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for i.i.d noise")
    
    args = parser.parse_args()
    process_dataset(args.input, args.output, args.alpha, args.seed)