from __future__ import annotations

from pathlib import Path

import splitfolders


INPUT_DIR = Path("data_raw")
OUTPUT_DIR = Path("data")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def split_dataset() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    splitfolders.ratio(
        str(INPUT_DIR),
        output=str(OUTPUT_DIR),
        seed=SEED,
        ratio=(TRAIN_RATIO, VAL_RATIO, TEST_RATIO),
    )

    print("Dataset split completed.")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Ratio:  train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")


def main() -> None:
    split_dataset()


if __name__ == "__main__":
    main()