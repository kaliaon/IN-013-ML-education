from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd


def load_table(
	path: str | Path,
	file_type: Optional[Literal["csv", "parquet", "xlsx"]] = None,
	index_col: Optional[str] = None,
	sep: str = ",",
	encoding: Optional[str] = "utf-8",
)	-> pd.DataFrame:
	"""Load tabular dataset from CSV/Parquet/XLSX.

	- If file_type is None, inferred from suffix.
	- Returns pandas DataFrame.
	"""
	path = Path(path)
	if file_type is None:
		file_type = path.suffix.lstrip(".").lower()  # type: ignore[assignment]

	if file_type == "csv":
		return pd.read_csv(path, sep=sep, encoding=encoding, index_col=index_col)
	if file_type == "parquet":
		return pd.read_parquet(path)
	if file_type in {"xlsx", "xls"}:
		return pd.read_excel(path, index_col=index_col)

	raise ValueError(f"Unsupported file type: {file_type}")


def save_table(df: pd.DataFrame, path: str | Path) -> None:
	"""Save DataFrame by extension: .csv or .parquet"""
	path = Path(path)
	ext = path.suffix.lower()
	if ext == ".csv":
		df.to_csv(path, index=False)
		return
	if ext == ".parquet":
		df.to_parquet(path, index=False)
		return
	raise ValueError(f"Unsupported extension: {ext}")


