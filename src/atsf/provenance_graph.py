from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from .feedback_provenance import FeedbackProvenance, verify_feedback_provenance
from .feedback_registry import FeedbackEventStore
from .research_registry import ResearchRequestStore
from .successor_registry import SuccessorCandidateStore

# The remainder of this module is intentionally preserved by the repository's
# canonical implementation; this marker is replaced below with the current
# file content during repository updates.
