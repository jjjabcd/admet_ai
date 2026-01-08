from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
from chemprop.train import predict
from chemprop.utils import load_checkpoint, load_scalers
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class FoldEnsembleScores:
    """
    Args:
        dataset_type (Literal["regression", "classification"]): Dataset type.
        fold_idx (int): Fold index (1-based).
        ensemble_size (int): Number of ensemble members.
        metrics (Dict[str, Dict[str, float]]): metric -> {"mean": ..., "std": ...}

    Returns:
        None
    """
    dataset_type: Literal["regression", "classification"]
    fold_idx: int
    ensemble_size: int
    metrics: Dict[str, Dict[str, float]]


REGRESSION_METRICS: List[str] = ["MAE", "RMSE", "R2", "PCC"]
CLASSIFICATION_METRICS: List[str] = ["f1-score", "precision", "recall", "auc", "auc-roc"]


def _safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Args:
        y_true (np.ndarray): Ground truth.
        y_pred (np.ndarray): Prediction.

    Returns:
        float: PCC or NaN.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(pearsonr(y_true[mask], y_pred[mask])[0])


def _load_test_xy(
    test_csv_path: Path,
    smiles_col: str,
    label_col: str,
) -> Tuple[List[str], np.ndarray]:
    """
    Args:
        test_csv_path (Path): Path to fold test.csv.
        smiles_col (str): SMILES column name.
        label_col (str): Label column name.

    Returns:
        Tuple[List[str], np.ndarray]: (smiles_list, y_true)
    """
    df = pd.read_csv(test_csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"{smiles_col} not found in {test_csv_path}. Columns={list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"{label_col} not found in {test_csv_path}. Columns={list(df.columns)}")

    smiles = df[smiles_col].astype(str).tolist()
    y_true = df[label_col].to_numpy(dtype=float)
    return smiles, y_true


def _build_test_dataloader(
    smiles: List[str],
    y_true: np.ndarray,
    num_workers: int,
    batch_size: int,
) -> MoleculeDataLoader:
    """
    Args:
        smiles (List[str]): SMILES list.
        y_true (np.ndarray): Ground truth array.
        num_workers (int): DataLoader workers.
        batch_size (int): Batch size.

    Returns:
        MoleculeDataLoader: Chemprop dataloader.
    """
    dataset = MoleculeDataset(
        [MoleculeDatapoint(smiles=[s], targets=[float(y)]) for s, y in zip(smiles, y_true)]
    )
    return MoleculeDataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        shuffle=False,
    )


def _infer_device(gpu: Optional[int], no_cuda: bool) -> torch.device:
    """
    Args:
        gpu (Optional[int]): GPU index.
        no_cuda (bool): Force CPU.

    Returns:
        torch.device: Torch device.
    """
    if no_cuda or (not torch.cuda.is_available()):
        return torch.device("cpu")
    if gpu is None:
        return torch.device("cuda")
    return torch.device(f"cuda:{int(gpu)}")


def _get_member_checkpoints(
    fold_out_dir: Path,
    ensemble_size: int,
) -> List[Path]:
    """
    Args:
        fold_out_dir (Path): e.g., results/.../fold1
        ensemble_size (int): Number of ensemble members.

    Returns:
        List[Path]: Checkpoint paths ordered by member index.
    """
    fold0_dir = fold_out_dir / "fold_0"
    ckpts: List[Path] = []
    for i in range(int(ensemble_size)):
        p = fold0_dir / f"model_{i}" / "model.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}")
        ckpts.append(p)
    return ckpts


def _unpack_scalers(
    scalers: Union[Tuple[Any, ...], Any]
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Args:
        scalers (Union[Tuple[Any, ...], Any]): Output of load_scalers(). It can be a tuple of varying length.

    Returns:
        Tuple[Optional[Any], Optional[Any]]: (target_scaler, features_scaler).
        If a scaler is not provided, it returns None for that scaler.
    """
    if scalers is None:
        return None, None

    # load_scalers might return:
    #   (target_scaler, features_scaler)
    #   (target_scaler, features_scaler, ...extra)
    #   (target_scaler,)
    if isinstance(scalers, tuple):
        if len(scalers) == 0:
            return None, None
        if len(scalers) == 1:
            return scalers[0], None
        return scalers[0], scalers[1]

    # Some implementations may return a single scaler object
    return scalers, None


def _ensure_2d(preds: np.ndarray) -> np.ndarray:
    """
    Args:
        preds (np.ndarray): Predictions, possibly 1D.

    Returns:
        np.ndarray: Predictions as 2D array of shape (N, T).
    """
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    return preds


def _predict_member(
    checkpoint_path: Path,
    data_loader: MoleculeDataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Args:
        checkpoint_path (Path): Path to model.pt
        data_loader (MoleculeDataLoader): Data loader.
        device (torch.device): Torch device.

    Returns:
        np.ndarray: Predictions of shape (N, T).
    """
    model = load_checkpoint(path=str(checkpoint_path), device=device).eval()

    with torch.no_grad():
        preds = predict(model=model, data_loader=data_loader)

    preds = _ensure_2d(np.array(preds))

    # Inverse scaling if regression scaler exists
    scalers_out = load_scalers(path=str(checkpoint_path))
    scaler, _features_scaler = _unpack_scalers(scalers_out)

    if scaler is not None:
        # scaler.inverse_transform expects 2D array: (N, T)
        preds = scaler.inverse_transform(preds).astype(float)

    return preds.astype(float)


def _member_metrics_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Args:
        y_true (np.ndarray): Ground truth (N,).
        y_pred (np.ndarray): Prediction (N,).

    Returns:
        Dict[str, float]: Regression metrics.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return {k: float("nan") for k in REGRESSION_METRICS}

    yt = y_true[mask]
    yp = y_pred[mask]

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    pcc = _safe_pearsonr(yt, yp)

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "PCC": pcc}


def _member_metrics_classification(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Args:
        y_true (np.ndarray): Ground truth (N,). Expected 0/1.
        y_score (np.ndarray): Score/probability (N,).
        threshold (float): Threshold for y_score -> label.

    Returns:
        Dict[str, float]: Classification metrics.
            - 'auc-roc' = ROC-AUC
            - 'auc' = PR-AUC (Average Precision)
    """
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


def compute_fold_ensemble_scores(
    fold_out_dir: Path,
    test_csv_path: Path,
    dataset_type: Literal["regression", "classification"],
    smiles_col: str,
    label_col: str,
    ensemble_size: int,
    gpu: Optional[int],
    no_cuda: bool,
    batch_size: int,
    num_workers: int,
    cls_threshold: float = 0.5,
) -> FoldEnsembleScores:
    """
    Args:
        fold_out_dir (Path): e.g., results/.../fold1
        test_csv_path (Path): fold input test.csv
        dataset_type (Literal["regression","classification"]): Dataset type.
        smiles_col (str): SMILES column name in test.csv.
        label_col (str): Label column name in test.csv.
        ensemble_size (int): Number of ensemble members.
        gpu (Optional[int]): GPU index.
        no_cuda (bool): Force CPU.
        batch_size (int): Batch size for prediction.
        num_workers (int): DataLoader workers.
        cls_threshold (float): Threshold for classification metrics.

    Returns:
        FoldEnsembleScores: mean/std across ensemble members for each metric.
    """
    device = _infer_device(gpu=gpu, no_cuda=no_cuda)

    smiles, y_true = _load_test_xy(
        test_csv_path=test_csv_path,
        smiles_col=smiles_col,
        label_col=label_col,
    )
    data_loader = _build_test_dataloader(
        smiles=smiles,
        y_true=y_true,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    ckpts = _get_member_checkpoints(fold_out_dir=fold_out_dir, ensemble_size=ensemble_size)

    member_metrics: List[Dict[str, float]] = []
    for ckpt in ckpts:
        preds = _predict_member(checkpoint_path=ckpt, data_loader=data_loader, device=device)

        # Single-task expected
        if preds.ndim != 2 or preds.shape[1] < 1:
            raise ValueError(f"Unexpected pred shape from {ckpt}: {preds.shape}")

        y_hat = preds[:, 0].astype(float)

        if dataset_type == "regression":
            m = _member_metrics_regression(y_true=y_true, y_pred=y_hat)
        else:
            m = _member_metrics_classification(y_true=y_true, y_score=y_hat, threshold=cls_threshold)

        member_metrics.append(m)

    # Aggregate mean/std across members
    metrics_out: Dict[str, Dict[str, float]] = {}
    keys = REGRESSION_METRICS if dataset_type == "regression" else CLASSIFICATION_METRICS

    for k in keys:
        vals = np.array([mm.get(k, np.nan) for mm in member_metrics], dtype=float)
        metrics_out[k] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() >= 2 else float("nan"),
        }

    # fold_idx derived from folder name "foldX"
    fold_name = fold_out_dir.name
    fold_idx = int(fold_name.replace("fold", "")) if fold_name.startswith("fold") else -1

    return FoldEnsembleScores(
        dataset_type=dataset_type,
        fold_idx=fold_idx,
        ensemble_size=int(ensemble_size),
        metrics=metrics_out,
    )


def write_fold_test_scores_csv(
    fold_out_dir: Path,
    scores: FoldEnsembleScores,
    output_name: str = "test_scores.csv",
) -> Path:
    """
    Args:
        fold_out_dir (Path): e.g., results/.../fold1
        scores (FoldEnsembleScores): Ensemble scores.
        output_name (str): Output filename.

    Returns:
        Path: Saved CSV path.
    """
    row: Dict[str, float] = {}
    for metric_name, stats in scores.metrics.items():
        row[f"{metric_name}_mean"] = float(stats["mean"])
        row[f"{metric_name}_std"] = float(stats["std"])

    df = pd.DataFrame([row])
    out_path = (fold_out_dir / output_name).expanduser().resolve()
    df.to_csv(out_path, index=False)
    return out_path


def write_fold0_test_scores_json(
    fold_out_dir: Path,
    dataset_name: str,
    scores: FoldEnsembleScores,
    output_name: str = "test_scores.json",
) -> Path:
    """
    Args:
        fold_out_dir (Path): e.g., results/.../fold1
        dataset_name (str): Dataset name.
        scores (FoldEnsembleScores): Ensemble scores.
        output_name (str): Output filename under fold_0.

    Returns:
        Path: Saved JSON path.
    """
    fold0_dir = (fold_out_dir / "fold_0").expanduser().resolve()
    fold0_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset_name": dataset_name,
        "dataset_type": scores.dataset_type,
        "fold": int(scores.fold_idx),
        "ensemble_size": int(scores.ensemble_size),
        "metrics": scores.metrics,
    }

    out_path = fold0_dir / output_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_path
