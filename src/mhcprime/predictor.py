from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch

from .checkpointing import load_mhcprime_model
from .inference import run_mhcprime
from .preprocessing import prepare_input_dataframe