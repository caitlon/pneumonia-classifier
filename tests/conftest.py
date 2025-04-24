"""Configure pytest environment."""

import os
import sys
import types

# Disable CUDA before importing PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Disabling CUDA warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Configuring the simulation of the CPU version of PyTorch
if "GITHUB_ACTIONS" in os.environ or "CI" in os.environ:
    cuda_module = types.ModuleType("torch.cuda")
    cuda_module.is_available = lambda: False
    sys.modules['torch.cuda'] = cuda_module 