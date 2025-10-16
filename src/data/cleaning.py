from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def drop_empty_columns(df: pd.DataFrame, threshold_ratio: float = 0.98) -> pd.DataFrame:
	"""Drop columns where the share of missing values >= threshold_ratio."""
	missing_share = df.isna().mean()
	to_drop = missing_share[missing_share >= threshold_ratio].index.tolist()
	return df.drop(columns=to_drop)


def fill_missing_values(
	df: pd.DataFrame,
	numeric_strategy: str = "median",
	categorical_fill_value: str = "missing",
)	-> pd.DataFrame:
	"""Fill missing values for numeric and categorical columns."""
	df = df.copy()
	numeric_cols = df.select_dtypes(include=[np.number]).columns
	cat_cols = df.select_dtypes(exclude=[np.number]).columns

	if numeric_strategy == "median":
		for col in numeric_cols:
			df[col] = df[col].fillna(df[col].median())
	elif numeric_strategy == "mean":
		for col in numeric_cols:
			df[col] = df[col].fillna(df[col].mean())
	else:
		raise ValueError("numeric_strategy must be 'median' or 'mean'")

	for col in cat_cols:
		df[col] = df[col].fillna(categorical_fill_value)

	return df


def normalize_numeric(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
	"""Z-score normalize selected numeric columns (or all numeric if None)."""
	df = df.copy()
	cols = list(columns) if columns is not None else df.select_dtypes(include=[np.number]).columns.tolist()
	for col in cols:
		std = df[col].std()
		if std and std > 0:
			df[col] = (df[col] - df[col].mean()) / std
	return df


