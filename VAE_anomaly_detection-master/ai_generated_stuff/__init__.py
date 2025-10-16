"""Setup imports for ai_generated_stuff"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from model/
parent_dir = Path(__file__).parent.parent.resolve()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
