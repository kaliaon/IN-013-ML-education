from __future__ import annotations

from typing import Dict, Literal

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_proba=None, average: Literal["macro", "weighted"] = "weighted") -> Dict[str, float]:
	metrics = {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"f1": float(f1_score(y_true, y_pred, average=average)),
		"precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
	}
	if y_proba is not None:
		try:
			metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
		except Exception:
			pass
	return metrics


