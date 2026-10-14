"""CLI entry point: python -m src.xarm_pipeline.safety.validate --trajectory X.csv"""
import sys

from .preprocess import main

if __name__ == "__main__":
    sys.exit(main())
