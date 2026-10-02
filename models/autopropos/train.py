"""Train AutoProPos on Grain images using the shared image-training engine."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.common.image_training.main import main

if __name__ == "__main__":
    main(method="autopropos")
