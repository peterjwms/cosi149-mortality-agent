import sys
from pathlib import Path

# Add the `lib` directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent / "t-patchGNN"))
sys.path.append(str(Path(__file__).resolve().parent / "t-patchGNN/tPatchGNN"))
sys.path.append(str(Path(__file__).resolve().parent / "t-patchGNN/tpatch_lib"))
print(sys.path)
import tpatch_lib
print(dir(tpatch_lib))
print(tpatch_lib.__file__)

# Test the import
# from lib import utils

print("utils imported successfully")