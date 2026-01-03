from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PreparedDataBundle:
    data_dir: Path
    ratings: pd.DataFrame
    movies: pd.DataFrame
    tags: pd.DataFrame
    users: pd.DataFrame
    manifest: dict

