import sys
from pathlib import Path

# Make `import src.xarm_pipeline...` work when running plain `pytest`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
