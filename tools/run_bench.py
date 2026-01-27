import os
import sys
from pathlib import Path

# Workaround for Intel OpenMP duplicate runtime on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure project root is on sys.path so `import src` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.experiments import quick_zero_shot_benchmark  # noqa: E402


def main() -> None:
    os.makedirs("results", exist_ok=True)
    print("cuda", torch.cuda.is_available())

    # Keep this small so it finishes quickly on Windows.
    DATASETS = ["MATR", "CALCE", "HUST"]
    EPOCHS = 1

    print("=== A) sequence + ProtocolInvariantTransform ===")
    quick_zero_shot_benchmark(
        datasets=DATASETS,
        models=["lstm"],
        input_type="sequence",
        epochs=EPOCHS,
        use_normalization=True,
        processed_dir="data/processed",
        output_dir="results",
    )

    print("=== B) sequence + no normalization ===")
    quick_zero_shot_benchmark(
        datasets=DATASETS,
        models=["lstm"],
        input_type="sequence",
        epochs=EPOCHS,
        use_normalization=False,
        processed_dir="data/processed",
        output_dir="results",
    )

    print("=== C) full_sequence (built-in minmax) ===")
    quick_zero_shot_benchmark(
        datasets=DATASETS,
        models=["lstm"],
        input_type="full_sequence",
        epochs=EPOCHS,
        use_normalization=False,
        processed_dir="data/processed",
        output_dir="results",
    )


if __name__ == "__main__":
    main()

