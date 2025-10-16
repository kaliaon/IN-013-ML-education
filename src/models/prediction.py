from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
	accuracy_score,
	f1_score,
	mean_absolute_error,
	mean_squared_error,
	r2_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


TaskType = Literal["classification", "regression"]


@dataclass
class TrainResult:
	model_name: str
	best_params: Dict[str, object]
	metrics: Dict[str, float]
	pipeline: Pipeline


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
	numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
	categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

	numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
	categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

	return ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_features),
			("cat", categorical_transformer, categorical_features),
		]
	)


def train_model(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_val: Optional[pd.DataFrame] = None,
	y_val: Optional[pd.Series] = None,
	model: Literal["rf", "xgb", "lgbm"] = "rf",
	task: TaskType = "classification",
)	-> TrainResult:
	preprocessor = _build_preprocessor(X_train)

	if task == "classification":
		estimator_map = {
			"rf": RandomForestClassifier(n_estimators=300, random_state=42),
			"xgb": XGBClassifier(
				max_depth=6,
				n_estimators=400,
				learning_rate=0.05,
				subsample=0.9,
				colsample_bytree=0.9,
				eval_metric="logloss",
				n_jobs=4,
				random_state=42,
			),
			"lgbm": LGBMClassifier(n_estimators=400, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42),
		}
		pipeline = Pipeline(steps=[("preprocess", preprocessor), ("estimator", estimator_map[model])])
		pipeline.fit(X_train, y_train)
		metrics: Dict[str, float] = {}
		if X_val is not None and y_val is not None:
			pred = pipeline.predict(X_val)
			metrics["accuracy"] = float(accuracy_score(y_val, pred))
			metrics["f1"] = float(f1_score(y_val, pred, average="weighted"))
	else:
		estimator_map = {
			"rf": RandomForestRegressor(n_estimators=400, random_state=42),
			"xgb": XGBRegressor(
				max_depth=6,
				n_estimators=500,
				learning_rate=0.05,
				subsample=0.9,
				colsample_bytree=0.9,
				n_jobs=4,
				random_state=42,
			),
			"lgbm": LGBMRegressor(n_estimators=500, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42),
		}
		pipeline = Pipeline(steps=[("preprocess", preprocessor), ("estimator", estimator_map[model])])
		pipeline.fit(X_train, y_train)
		metrics = {}
		if X_val is not None and y_val is not None:
			pred = pipeline.predict(X_val)
			metrics["mae"] = float(mean_absolute_error(y_val, pred))
			metrics["rmse"] = float(mean_squared_error(y_val, pred, squared=False))
			metrics["r2"] = float(r2_score(y_val, pred))

	return TrainResult(model_name=model, best_params=pipeline.named_steps["estimator"].get_params(), metrics=metrics, pipeline=pipeline)


