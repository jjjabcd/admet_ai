from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

REGRESSION_METRICS: List[str] = ["MAE", "RMSE", "R2", "PCC"]
CLASSIFICATION_METRICS: List[str] = ["f1-score", "precision", "recall", "auc", "auc-roc"]


def _safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: PCC or NaN.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(pearsonr(y_true[mask], y_pred[mask])[0])


def compute_regression_metrics(test_preds_path: Path) -> Dict[str, float]:
    """
    Expects overwritten test_preds.csv with columns: smiles,y_true,y_pred

    Args:
        test_preds_path (Path): Path to overwritten test_preds.csv.

    Returns:
        Dict[str, float]: {MAE, RMSE, R2, PCC}
    """
    df = pd.read_csv(test_preds_path)
    for col in ["y_true", "y_pred"]:
        if col not in df.columns:
            raise ValueError(f"{col} not found in {test_preds_path}. Columns={list(df.columns)}")

    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "PCC": float("nan")}

    mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
    r2 = float(r2_score(y_true[mask], y_pred[mask]))
    pcc = _safe_pearsonr(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "PCC": pcc}


def compute_classification_metrics(
    test_preds_path: Path,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Expects overwritten test_preds.csv with columns: smiles,y_true,y_score

    Args:
        test_preds_path (Path): Path to overwritten test_preds.csv.
        threshold (float): Threshold for y_score -> predicted label.

    Returns:
        Dict[str, float]: {f1-score, precision, recall, auc, auc-roc}
            - 'auc-roc' = ROC-AUC
            - 'auc' = PR-AUC (Average Precision)
    """
    df = pd.read_csv(test_preds_path)
    for col in ["y_true", "y_score"]:
        if col not in df.columns:
            raise ValueError(f"{col} not found in {test_preds_path}. Columns={list(df.columns)}")

    y_true = df["y_true"].to_numpy(dtype=float)
    y_score = df["y_score"].to_numpy(dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    if int(mask.sum()) == 0:
        return {k: float("nan") for k in CLASSIFICATION_METRICS}

    yt = y_true[mask].astype(int)
    ys = y_score[mask].astype(float)
    yp = (ys >= float(threshold)).astype(int)

    precision = float(precision_score(yt, yp, zero_division=0))
    recall = float(recall_score(yt, yp, zero_division=0))
    f1 = float(f1_score(yt, yp, zero_division=0))

    if len(np.unique(yt)) < 2:
        auc_roc = float("nan")
        auc_pr = float("nan")
    else:
        auc_roc = float(roc_auc_score(yt, ys))
        auc_pr = float(average_precision_score(yt, ys))

    return {"f1-score": f1, "precision": precision, "recall": recall, "auc": auc_pr, "auc-roc": auc_roc}


def save_metric_csv(
    dataset_out_dir: Path,
    fold_num: int,
    dataset_type: str,
    output_name: str = "metric.csv",
    cls_threshold: float = 0.5,
) -> Path:
    """
    Save fold-wise metrics and mean/std into one CSV at dataset_out_dir.

    Args:
        dataset_out_dir (Path): e.g., .../results/chemprop_rdkit/HIA_Hou
        fold_num (int): Number of folds (expects fold1 ... fold{fold_num})
        dataset_type (str): "regression" or "classification"
        output_name (str): Output filename
        cls_threshold (float): Threshold for classification label prediction

    Returns:
        Path: Saved metric.csv path
    """
    dataset_out_dir = dataset_out_dir.expanduser().resolve()
    if dataset_type not in ("regression", "classification"):
        raise ValueError(f"Invalid dataset_type={dataset_type}")

    rows: List[Dict[str, float]] = []
    idxs: List[str] = []

    for fold_idx in range(1, fold_num + 1):
        fold_out = dataset_out_dir / f"fold{fold_idx}"
        test_preds_path = fold_out / "test_preds.csv"
        if not test_preds_path.exists():
            raise FileNotFoundError(test_preds_path)

        if dataset_type == "regression":
            m = compute_regression_metrics(test_preds_path)
            m = {k: m.get(k, float("nan")) for k in REGRESSION_METRICS}
        else:
            m = compute_classification_metrics(test_preds_path, threshold=cls_threshold)
            m = {k: m.get(k, float("nan")) for k in CLASSIFICATION_METRICS}

        rows.append(m)
        idxs.append(f"fold{fold_idx}")

    df = pd.DataFrame(rows, index=idxs)
    df.loc["mean"] = df.mean(axis=0, numeric_only=True)
    df.loc["std"] = df.std(axis=0, ddof=1, numeric_only=True)

    out_path = dataset_out_dir / output_name
    df.to_csv(out_path, index=True)
    return out_path
