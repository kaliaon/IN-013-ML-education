from __future__ import annotations

from typing import Dict, List

import pandas as pd


def summarize_results(results: List[Dict[str, float]]) -> pd.DataFrame:
	"""Convert list of metric dicts to a sorted DataFrame (by primary metric)."""
	df = pd.DataFrame(results).copy()
	primary = None
	for candidate in ["roc_auc", "f1", "accuracy", "r2", "rmse", "mae"]:
		if candidate in df.columns:
			primary = candidate
			break
	if primary is not None:
		ascending = primary in {"rmse", "mae"}
		df = df.sort_values(by=primary, ascending=ascending)
	return df.reset_index(drop=True)


