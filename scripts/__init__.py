import sys
from pathlib import Path

# add repository root to python path, so files in this folder can import from src folder
sys.path.insert(0, str(Path(__file__).parents[1]))
