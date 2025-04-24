"""Configure pytest environment."""

import os
import sys

# Disable CUDA before importing PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Disabling CUDA warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Configuring the simulation of the CPU version of PyTorch
if "GITHUB_ACTIONS" in os.environ or "CI" in os.environ:
    sys.modules['torch.cuda'] = type('', (), {'is_available': lambda: False}) 