import torch
import numpy as np

NP_FLOAT32_64 = np.float32 if torch.backends.mps.is_available() else np.float64
TH_FLOAT32_64 = torch.float32 if torch.backends.mps.is_available() else torch.float64