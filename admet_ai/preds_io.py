# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional

# import ast
# import numpy as np
# import pandas as pd


# @dataclass(frozen=True)
# class OverwriteResult:
#     """
#     Args:
#         path (Path): Saved file path.
#         y_pred_found (bool): Whether a valid prediction/score column was found and written.
#         backup_path (Optional[Path]): Backup path if backup was enabled; otherwise None.

#     Returns:
#         None
#     """
#     path: Path
#     y_pred_found: bool
#     backup_path: Optional[Path]


# def normalize_smiles_value(x: object) -> str:
#     """
#     Args:
#         x (object): Raw SMILES cell value that may look like "['CCO']" or "CCO".

#     Returns:
#         str: Normalized SMILES string (best-effort).
#     """
#     if x is None or (isinstance(x, float) and np.isnan(x)):
#         return ""

#     s = str(x).strip()

#     # Case 1: stringified list like "['SMILES']"
#     if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
#         try:
#             obj = ast.literal_eval(s)
#             if isinstance(obj, list) and len(obj) > 0:
#                 return str(obj[0]).strip()
#         except Exception:
#             pass

#     # Case 2: already a plain string
#     # Also strip wrapping quotes if present
#     if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
#         s = s[1:-1].strip()

#     return s


# def _build_y_true_map(
#     test_csv_path: Path,
#     smiles_col: str,
#     label_col: str,
# ) -> dict[str, float]:
#     """
#     Args:
#         test_csv_path (Path): Path to fold test.csv.
#         smiles_col (str): SMILES column name.
#         label_col (str): Label column name.

#     Returns:
#         dict[str, float]: Mapping from normalized SMILES to y_true.
#     """
#     test_df = pd.read_csv(test_csv_path)

#     if smiles_col not in test_df.columns:
#         raise ValueError(f"{smiles_col} not found in {test_csv_path}. Columns={list(test_df.columns)}")
#     if label_col not in test_df.columns:
#         raise ValueError(f"{label_col} not found in {test_csv_path}. Columns={list(test_df.columns)}")

#     gt = test_df[[smiles_col, label_col]].drop_duplicates(subset=[smiles_col], keep="first").copy()
#     gt[smiles_col] = gt[smiles_col].map(normalize_smiles_value)

#     out: dict[str, float] = {}
#     for smi, y in zip(gt[smiles_col].astype(str), gt[label_col].astype(float)):
#         if smi != "":
#             out[smi] = float(y)
#     return out


# def _backup_then_overwrite(
#     path: Path,
#     backup: bool,
#     backup_suffix: str,
# ) -> Optional[Path]:
#     """
#     Args:
#         path (Path): Original file path to overwrite.
#         backup (bool): Whether to backup.
#         backup_suffix (str): Suffix for backup file.

#     Returns:
#         Optional[Path]: Backup path if created, else None.
#     """
#     if not backup:
#         return None

#     backup_path = path.with_name(path.name + backup_suffix)
#     if backup_path.exists():
#         return backup_path

#     # Move original to backup
#     path.replace(backup_path)
#     return backup_path


# def overwrite_test_preds_regression(
#     test_csv_path: Path,
#     chemprop_test_preds_path: Path,
#     label_col: str,
#     smiles_col: str = "smiles",
#     pred_col: Optional[str] = None,
#     backup: bool = True,
#     # backup_suffix: str = ".chemprop.bak",
#     target_only_tolerance: float = 1e-12,
# ) -> OverwriteResult:
#     """
#     Overwrite Chemprop test_preds.csv with standardized regression format:
#         smiles,y_true,y_pred

#     Args:
#         test_csv_path (Path): Fold test.csv (source of y_true).
#         chemprop_test_preds_path (Path): Chemprop-generated test_preds.csv.
#         label_col (str): Label column name in test.csv.
#         smiles_col (str): SMILES column name.
#         pred_col (Optional[str]): Force prediction column name in Chemprop file.
#         backup (bool): Keep original as *.chemprop.bak.
#         backup_suffix (str): Backup suffix.
#         target_only_tolerance (float): If candidate equals y_true within tolerance, treat as target-only dump.

#     Returns:
#         OverwriteResult: Result metadata.
#     """
#     test_csv_path = test_csv_path.expanduser().resolve()
#     chemprop_test_preds_path = chemprop_test_preds_path.expanduser().resolve()

#     if not test_csv_path.exists():
#         raise FileNotFoundError(test_csv_path)
#     if not chemprop_test_preds_path.exists():
#         raise FileNotFoundError(chemprop_test_preds_path)

#     preds_df = pd.read_csv(chemprop_test_preds_path)
#     if smiles_col not in preds_df.columns:
#         candidates = ["smiles", "SMILES", "Smiles", "Drug"]
#         found = None
#         for c in candidates:
#             if c in preds_df.columns:
#                 found = c
#                 break
#         if found is None:
#             raise ValueError(
#                 f"{smiles_col} not found in {chemprop_test_preds_path}. Columns={list(preds_df.columns)}"
#             )
#         smiles_col = found

#     # Normalize smiles in preds
#     out_df = pd.DataFrame({smiles_col: preds_df[smiles_col].map(normalize_smiles_value)})

#     y_true_map = _build_y_true_map(test_csv_path, smiles_col=smiles_col, label_col=label_col)
#     out_df["y_true"] = out_df[smiles_col].map(y_true_map).astype(float)

#     # Infer y_pred
#     y_pred_found = True
#     if pred_col is not None:
#         if pred_col not in preds_df.columns:
#             raise ValueError(f"pred_col='{pred_col}' not found. Columns={list(preds_df.columns)}")
#         cand = pd.to_numeric(preds_df[pred_col], errors="coerce").astype(float)
#     else:
#         numeric_cols = [
#             c for c in preds_df.columns
#             if c != smiles_col and pd.api.types.is_numeric_dtype(preds_df[c])
#         ]
#         if len(numeric_cols) == 0:
#             y_pred_found = False
#             cand = pd.Series([np.nan] * len(out_df))
#         else:
#             cand = pd.to_numeric(preds_df[numeric_cols[0]], errors="coerce").astype(float)

#     # Detect target-only dump
#     y_true_arr = out_df["y_true"].to_numpy(dtype=float)
#     cand_arr = cand.to_numpy(dtype=float)
#     mask = np.isfinite(y_true_arr) & np.isfinite(cand_arr)

#     if mask.sum() >= 5:
#         diff = np.abs(y_true_arr[mask] - cand_arr[mask])
#         if float(np.nanmax(diff)) <= float(target_only_tolerance):
#             y_pred_found = False
#             cand = pd.Series([np.nan] * len(out_df))

#     out_df["y_pred"] = pd.to_numeric(cand, errors="coerce").astype(float)

#     # Backup then overwrite
#     backup_path = _backup_then_overwrite(chemprop_test_preds_path, backup=backup, backup_suffix=backup_suffix)
#     chemprop_test_preds_path.parent.mkdir(parents=True, exist_ok=True)
#     out_df.to_csv(chemprop_test_preds_path, index=False)

#     return OverwriteResult(path=chemprop_test_preds_path, y_pred_found=y_pred_found, backup_path=backup_path)


# def overwrite_test_preds_classification(
#     test_csv_path: Path,
#     chemprop_test_preds_path: Path,
#     label_col: str,
#     smiles_col: str = "smiles",
#     score_col: Optional[str] = None,
#     backup: bool = True,
#     backup_suffix: str = ".chemprop.bak",
# ) -> OverwriteResult:
#     """
#     Overwrite Chemprop test_preds.csv with standardized classification format:
#         smiles,y_true,y_score

#     Args:
#         test_csv_path (Path): Fold test.csv (source of y_true).
#         chemprop_test_preds_path (Path): Chemprop-generated test_preds.csv.
#         label_col (str): Label column name in test.csv (0/1 expected).
#         smiles_col (str): SMILES column name.
#         score_col (Optional[str]): Force score column name in Chemprop file.
#         backup (bool): Keep original as *.chemprop.bak.
#         backup_suffix (str): Backup suffix.

#     Returns:
#         OverwriteResult: Result metadata.
#     """
#     test_csv_path = test_csv_path.expanduser().resolve()
#     chemprop_test_preds_path = chemprop_test_preds_path.expanduser().resolve()

#     if not test_csv_path.exists():
#         raise FileNotFoundError(test_csv_path)
#     if not chemprop_test_preds_path.exists():
#         raise FileNotFoundError(chemprop_test_preds_path)

#     preds_df = pd.read_csv(chemprop_test_preds_path)
#     if smiles_col not in preds_df.columns:
#         raise ValueError(f"{smiles_col} not found in {chemprop_test_preds_path}. Columns={list(preds_df.columns)}")

#     out_df = pd.DataFrame({smiles_col: preds_df[smiles_col].map(normalize_smiles_value)})

#     y_true_map = _build_y_true_map(test_csv_path, smiles_col=smiles_col, label_col=label_col)
#     out_df["y_true"] = out_df[smiles_col].map(y_true_map).astype(float)

#     # Infer y_score
#     if score_col is not None:
#         if score_col not in preds_df.columns:
#             raise ValueError(f"score_col='{score_col}' not found. Columns={list(preds_df.columns)}")
#         y_score = preds_df[score_col]
#     else:
#         numeric_cols = [
#             c for c in preds_df.columns
#             if c != smiles_col and pd.api.types.is_numeric_dtype(preds_df[c])
#         ]
#         if len(numeric_cols) == 0:
#             raise ValueError(f"No numeric score column found. Columns={list(preds_df.columns)}")
#         y_score = preds_df[numeric_cols[0]]

#     out_df["y_score"] = pd.to_numeric(y_score, errors="coerce").astype(float)

#     backup_path = _backup_then_overwrite(chemprop_test_preds_path, backup=backup, backup_suffix=backup_suffix)
#     chemprop_test_preds_path.parent.mkdir(parents=True, exist_ok=True)
#     out_df.to_csv(chemprop_test_preds_path, index=False)

#     return OverwriteResult(path=chemprop_test_preds_path, y_pred_found=True, backup_path=backup_path)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OverwriteResult:
    """
    Args:
        y_pred_found (bool): Whether y_pred column was found in Chemprop preds.
        backup_path (Optional[Path]): Backup path if created.

    Returns:
        None
    """
    y_pred_found: bool
    backup_path: Optional[Path]


def _normalize_smiles_cell(x: str) -> str:
    """
    Args:
        x (str): Raw SMILES cell string.

    Returns:
        str: Normalized SMILES string.
    """
    s = str(x).strip()

    # Handle Chemprop-like weird dumps: "['CCO']"
    if s.startswith("['") and s.endswith("']"):
        s = s[2:-2].strip()
    if s.startswith('["') and s.endswith('"]'):
        s = s[2:-2].strip()

    # Strip extra quotes
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1].strip()

    return s


def _build_y_true_map(
    test_csv_path: Path,
    smiles_col: str,
    label_col: str,
) -> Dict[str, float]:
    """
    Args:
        test_csv_path (Path): Original test CSV containing y_true.
        smiles_col (str): SMILES column name in test_csv_path.
        label_col (str): Label column name in test_csv_path.

    Returns:
        Dict[str, float]: normalized_smiles -> y_true
    """
    test_df = pd.read_csv(test_csv_path)

    if smiles_col not in test_df.columns:
        raise ValueError(f"{smiles_col} not found in {test_csv_path}. Columns={list(test_df.columns)}")
    if label_col not in test_df.columns:
        raise ValueError(f"{label_col} not found in {test_csv_path}. Columns={list(test_df.columns)}")

    smi = test_df[smiles_col].astype(str).map(_normalize_smiles_cell)
    y = pd.to_numeric(test_df[label_col], errors="coerce")

    # If duplicates exist, keep first
    tmp = pd.DataFrame({"smiles": smi, "y_true": y}).drop_duplicates(subset=["smiles"], keep="first")
    return dict(zip(tmp["smiles"].tolist(), tmp["y_true"].astype(float).tolist()))


def _infer_pred_col(preds_df: pd.DataFrame, label_col: str) -> Optional[str]:
    """
    Args:
        preds_df (pd.DataFrame): Chemprop test_preds.csv loaded dataframe.
        label_col (str): Label/task name.

    Returns:
        Optional[str]: prediction column name, if found.
    """
    preferred = [f"{label_col}_pred", label_col, "pred", "prediction", "preds"]
    for c in preferred:
        if c in preds_df.columns:
            return c

    # Any numeric col excluding smiles-like columns
    smiles_like = {"smiles", "SMILES", "Smiles", "Drug"}
    numeric_cols = [
        c for c in preds_df.columns
        if c not in smiles_like and pd.api.types.is_numeric_dtype(preds_df[c])
    ]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    if len(numeric_cols) > 1:
        # Heuristic: pick the first
        return numeric_cols[0]
    return None


def overwrite_test_preds_regression(
    test_csv_path: Path,
    chemprop_test_preds_path: Path,
    label_col: str,
    input_smiles_col: str,
    preds_smiles_col: str = "smiles",
    pred_col: Optional[str] = None,
    backup: bool = True,
) -> OverwriteResult:
    """
    Args:
        test_csv_path (Path): Original fold test CSV containing y_true (SMILES column is input_smiles_col).
        chemprop_test_preds_path (Path): Chemprop test_preds.csv containing y_pred (SMILES column is preds_smiles_col).
        label_col (str): Label/task name in test_csv_path (y_true) and usually also in chemprop preds.
        input_smiles_col (str): SMILES column name in test_csv_path (e.g., "SMILES").
        preds_smiles_col (str): SMILES column name in chemprop_test_preds_path (usually "smiles").
        pred_col (Optional[str]): Prediction column name in chemprop_test_preds_path; if None inferred.
        backup (bool): If True, create *.chemprop.bak before overwriting.

    Returns:
        OverwriteResult
    """
    test_csv_path = Path(test_csv_path)
    chemprop_test_preds_path = Path(chemprop_test_preds_path)

    preds_df = pd.read_csv(chemprop_test_preds_path)

    # Validate preds SMILES column (Chemprop output)
    if preds_smiles_col not in preds_df.columns:
        # try common alternatives
        alt = None
        for c in ["smiles", "SMILES", "Smiles", "Drug"]:
            if c in preds_df.columns:
                alt = c
                break
        if alt is None:
            raise ValueError(
                f"{preds_smiles_col} not found in {chemprop_test_preds_path}. Columns={list(preds_df.columns)}"
            )
        preds_smiles_col = alt

    # Build y_true map from input test.csv using input_smiles_col (THIS FIXES YOUR ERROR)
    y_true_map = _build_y_true_map(test_csv_path, smiles_col=input_smiles_col, label_col=label_col)

    # Infer y_pred column from preds_df
    if pred_col is None:
        pred_col = _infer_pred_col(preds_df, label_col=label_col)

    y_pred_found = True
    if pred_col is None or pred_col not in preds_df.columns:
        y_pred_found = False
        y_pred = pd.Series([np.nan] * len(preds_df))
    else:
        y_pred = pd.to_numeric(preds_df[pred_col], errors="coerce")

    # Normalize preds smiles and map y_true
    smiles_norm = preds_df[preds_smiles_col].astype(str).map(_normalize_smiles_cell)
    y_true = smiles_norm.map(y_true_map)

    out_df = pd.DataFrame(
        {
            "smiles": smiles_norm,
            "y_true": pd.to_numeric(y_true, errors="coerce"),
            "y_pred": y_pred,
        }
    )

    backup_path: Optional[Path] = None
    if backup:
        backup_path = chemprop_test_preds_path.with_suffix(chemprop_test_preds_path.suffix + ".chemprop.bak")
        if not backup_path.exists():
            chemprop_test_preds_path.replace(backup_path)
        else:
            # If backup exists, keep it and proceed to overwrite original path directly
            pass

        # If we moved original to backup, we must write new file to original path
        out_df.to_csv(chemprop_test_preds_path, index=False)
    else:
        out_df.to_csv(chemprop_test_preds_path, index=False)

    return OverwriteResult(y_pred_found=y_pred_found, backup_path=backup_path)
