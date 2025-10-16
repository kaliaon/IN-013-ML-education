from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap


def compute_permutation_importance(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5, random_state: int = 42) -> pd.DataFrame:
	"""Compute permutation importance for fitted model (sklearn-compatible)."""
	from sklearn.inspection import permutation_importance
	result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
	importances = pd.DataFrame({"feature": X.columns, "importance_mean": result.importances_mean, "importance_std": result.importances_std})
	return importances.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def compute_shap_values(pipeline, X_sample: pd.DataFrame, max_samples: int = 200) -> shap._explanation.Explanation:
	"""Compute SHAP values for a fitted pipeline. Uses TreeExplainer when possible."""
	X_use = X_sample.sample(n=min(max_samples, len(X_sample)), random_state=42)
	estimator = pipeline.named_steps["estimator"]
	try:
		explainer = shap.TreeExplainer(estimator)
		return explainer(X_use)
	except Exception:
		explainer = shap.Explainer(estimator.predict, X_use)
		return explainer(X_use)


