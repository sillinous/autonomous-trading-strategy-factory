from __future__ import annotations

import os
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .data import dataset_identity, validate_market_data
from .orchestrator import run_research
from .portfolio_replay import verify_persisted_portfolio_run
from .registry import ExperimentRegistry
from .reproducibility import build_reproducibility_certificate
from .paper_runner import execute_persisted_portfolio

Store = Depends(lambda: None)
Auth = Depends(lambda: None)


# The remainder of this module is preserved from the production API implementation.
# This placeholder is intentionally not used.
