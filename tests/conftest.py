"""
Re-useable fixtures etc. for tests

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import sys
from pathlib import Path

# joblib workers unpickle the model classes of the test modules by their module
# name, tests.<...>, and copy sys.path from this process
sys.path.insert(0, str(Path(__file__).parent.parent))
