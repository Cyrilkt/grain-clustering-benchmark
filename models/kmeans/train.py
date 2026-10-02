"""Run the native PyTorch KMeans on the frozen Grain embeddings."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from grain_benchmark.cli import main

if __name__ == "__main__":
    main(["kmeans", *sys.argv[1:]])
