from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResult:
	labels: np.ndarray
	silhouette: Optional[float]
	model_name: str
	model_params: Dict[str, object]


def _scale_numeric(X: pd.DataFrame) -> pd.DataFrame:
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))
	X_num = pd.DataFrame(X_scaled, columns=X.select_dtypes(include=[np.number]).columns, index=X.index)
	X_other = X.drop(columns=X_num.columns)
	return pd.concat([X_num, X_other], axis=1)


def kmeans_cluster(
	X: pd.DataFrame,
	n_clusters: int = 5,
	random_state: int = 42,
	mini_batch: bool = False,
	compute_silhouette: bool = True,
)	-> ClusteringResult:
	X_scaled = _scale_numeric(X)
	model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state) if mini_batch else KMeans(
		n_clusters=n_clusters,
		random_state=random_state,
		n_init="auto",
	)
	labels = model.fit_predict(X_scaled.select_dtypes(include=[np.number]))
	s = silhouette_score(X_scaled.select_dtypes(include=[np.number]), labels) if compute_silhouette and len(set(labels)) > 1 else None
	return ClusteringResult(labels=np.asarray(labels), silhouette=s, model_name=model.__class__.__name__, model_params=model.get_params())


def dbscan_cluster(
	X: pd.DataFrame,
	eps: float = 0.5,
	min_samples: int = 5,
	compute_silhouette: bool = True,
)	-> ClusteringResult:
	X_scaled = _scale_numeric(X)
	model = DBSCAN(eps=eps, min_samples=min_samples)
	labels = model.fit_predict(X_scaled.select_dtypes(include=[np.number]))
	# silhouette undefined for single cluster or when all points are noise (-1)
	valid_labels = set(labels) - {-1}
	s = silhouette_score(X_scaled.select_dtypes(include=[np.number]), labels) if compute_silhouette and len(valid_labels) > 1 else None
	return ClusteringResult(labels=np.asarray(labels), silhouette=s, model_name="DBSCAN", model_params=model.get_params())


