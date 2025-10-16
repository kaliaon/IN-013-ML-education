from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_xy(
	df: pd.DataFrame,
	target_column: str,
	test_size: float = 0.2,
	val_size: float = 0.1,
	random_state: int = 42,
)	-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
	"""Split a DataFrame into train/val/test and return X_train, X_val, y_train, y_val, X_test, y_test.

	Validation split is taken from the train portion to preserve test_size fraction.
	"""
	X = df.drop(columns=[target_column])
	y = df[target_column]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=None
	)

	val_relative = val_size / (1.0 - test_size)
	X_train, X_val, y_train, y_val = train_test_split(
		X_train, y_train, test_size=val_relative, random_state=random_state, stratify=None
	)

	return X_train, X_val, y_train, y_val, X_test, y_test


