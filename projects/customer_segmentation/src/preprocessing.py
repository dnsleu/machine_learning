from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

@dataclass(frozen=True)
class PrepConfig:
    feature_cols: list[str]
    category_cols: list[str] | None = None
    rename_map: dict[str, str] | None = None

def clean_columns_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip white spaces and normalize punctuation / spaces.
    """
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w_]", "", regex=True)
    )
    return out

def apply_column_renames(df: pd.DataFrame, rename_map: Optional[dict[str, str]]) -> pd.DataFrame:
    """
    Rename specific columns using a dictionary mapping
    """
    if not rename_map:
        return df
    return df.rename(columns=rename_map)

def build_model_table(df: pd.DataFrame, cfg: PrepConfig) -> pd.DataFrame:
    """
    Select features, convert to numeric, drop missing rows
    """
    out = df.copy()

    # Check if feature columns available
    missing = [c for c in cfg.feature_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Available: {list(out.columns)}")
    
    keep_cols: list[str] = list(cfg.feature_cols)
    if cfg.category_cols:
        keep_cols += [c for c in cfg.category_cols if c in out.columns]

    model_df = out[keep_cols].copy()

    # Convert feature columns to numeric
    for c in cfg.feature_cols:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

    # Drop rows with missing values
    model_df = model_df.dropna(subset=cfg.feature_cols).reset_index(drop=True)
    return model_df

def save_processed(model_df: pd.DataFrame, out_path: Path) -> Path:
    """
    Save processed dataframe
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(out_path, index=False)
    return out_path