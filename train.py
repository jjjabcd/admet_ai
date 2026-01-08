from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

from scripts.tdc_constants import DATASET_TO_TYPE, DATASET_TYPE_TO_METRICS_COMMAND_LINE

from admet_ai.preds_io import (
    overwrite_test_preds_classification,
    overwrite_test_preds_regression,
)

from admet_ai.ensemble_eval import (
    compute_fold_ensemble_scores,
    write_fold_test_scores_csv,
    write_fold0_test_scores_json
)

from admet_ai.metric import save_metric_csv

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def train_dataset_dir(
    dataset_dir: Path,
    output_dir: Path,
    fold_num: int,
    dataset_name: Optional[str],
    dataset_type: Optional[str],
    compound_col: str,
    label_col: str,
    ensemble_size: int,
    model_type: str,
    gpu: Optional[int],
    no_cuda: bool,
    batch_size: Optional[int],
    metric: Optional[str],
    extra_metrics: Optional[List[str]],
    quiet: bool,
    save_preds: bool,
    dry_run: bool,
    overwrite_backup: bool,
) -> None:
    """
    Args:
        dataset_dir (Path): Root dir containing fold{idx}/train|val|test.csv
        output_dir (Path): Root output dir
        fold_num (int): Number of folds (expects fold1 ... fold{fold_num})
        dataset_name (Optional[str]): Dataset name used for output folder naming
        dataset_type (Optional[str]): "regression" or "classification"; if None, inferred from DATASET_TO_TYPE
        compound_col (str): SMILES column name
        label_col (str): Label column name (varies per dataset)
        ensemble_size (int): Chemprop ensemble size (D-MPNN ensemble)
        model_type (str): Output subfolder name (e.g., "chemprop_rdkit")
        gpu (Optional[int]): GPU index passed to Chemprop (--gpu)
        no_cuda (bool): Force CPU for Chemprop (--no_cuda)
        batch_size (Optional[int]): Batch size passed to Chemprop (--batch_size)
        metric (Optional[str]): Primary metric passed to Chemprop (--metric)
        extra_metrics (Optional[List[str]]): Extra metrics passed to Chemprop (--extra_metrics ...)
        quiet (bool): Pass --quiet to Chemprop
        save_preds (bool): Pass --save_preds to Chemprop
        dry_run (bool): Print commands only, do not run
        overwrite_backup (bool): If True, keep Chemprop original test_preds.csv as *.chemprop.bak

    Returns:
        None
    """
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    dataset_name = dataset_name or dataset_dir.name

    # Infer dataset_type if not provided
    if dataset_type is None:
        if dataset_name not in DATASET_TO_TYPE:
            raise ValueError(
                f"dataset_type must be provided explicitly for dataset '{dataset_name}'"
            )
        dataset_type = DATASET_TO_TYPE[dataset_name]

    if dataset_type not in ("regression", "classification"):
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    if fold_num < 1:
        raise ValueError(f"fold_num must be >= 1, got {fold_num}")

    if dataset_type not in DATASET_TYPE_TO_METRICS_COMMAND_LINE:
        raise ValueError(
            f"No default metric args found for dataset_type='{dataset_type}'"
        )
    default_metric_args = DATASET_TYPE_TO_METRICS_COMMAND_LINE[dataset_type]

    # For ensemble evaluation (post-hoc)
    # Chemprop default is often 8 workers when GPU is available; keep a safe fixed default.
    num_workers_eval = 8
    batch_size_eval = int(batch_size) if batch_size is not None else 50

    # -----------------------------
    # Train folds
    # -----------------------------
    for fold_idx in range(1, fold_num + 1):
        fold_in_dir = dataset_dir / f"fold{fold_idx}"
        train_csv = fold_in_dir / "train.csv"
        val_csv = fold_in_dir / "val.csv"
        test_csv = fold_in_dir / "test.csv"

        for p in (train_csv, val_csv, test_csv):
            if not p.exists():
                raise FileNotFoundError(p)

        fold_out_dir = output_dir / model_type / dataset_name / f"fold{fold_idx}"
        fold_out_dir.mkdir(parents=True, exist_ok=True)

        command: List[str] = [
            "chemprop_train",
            "--data_path",
            str(train_csv),
            "--separate_val_path",
            str(val_csv),
            "--separate_test_path",
            str(test_csv),
            "--dataset_type",
            dataset_type,
            "--smiles_column",
            compound_col,
            "--target_columns",
            label_col,
            "--ensemble_size",
            str(ensemble_size),
            "--save_dir",
            str(fold_out_dir),
        ]

        # Metric control (Chemprop-supported only)
        user_extra_metrics = extra_metrics or []
        if metric is None and len(user_extra_metrics) == 0:
            command += list(default_metric_args)
        else:
            if metric is not None:
                command += ["--metric", metric]
            if len(user_extra_metrics) > 0:
                command += ["--extra_metrics", *user_extra_metrics]

        # Runtime controls
        if batch_size is not None:
            if batch_size < 1:
                raise ValueError(f"batch_size must be >= 1, got {batch_size}")
            command += ["--batch_size", str(batch_size)]

        # Device controls
        if no_cuda:
            command += ["--no_cuda"]
        elif gpu is not None:
            command += ["--gpu", str(gpu)]

        # Outputs
        if save_preds:
            command += ["--save_preds"]
        if quiet:
            command += ["--quiet"]

        print(f"\n[INFO] Fold {fold_idx}/{fold_num}")
        print(" ".join(command))

        if dry_run:
            continue

        # -----------------------------
        # 1) Train (Chemprop)
        # -----------------------------
        subprocess.run(command, check=True)

        # -----------------------------
        # 2) Overwrite foldX/test_preds.csv (standardized)
        # -----------------------------
        chemprop_test_preds_path = fold_out_dir / "test_preds.csv"
        if not chemprop_test_preds_path.exists():
            raise FileNotFoundError(
                f"Chemprop did not create test_preds.csv at: {chemprop_test_preds_path}"
            )

        if dataset_type == "regression":
            res = overwrite_test_preds_regression(
                test_csv_path=test_csv,
                chemprop_test_preds_path=chemprop_test_preds_path,
                label_col=label_col,
                smiles_col=compound_col,
                pred_col=None,
                backup=overwrite_backup,
            )
            print(
                f"[INFO] Overwrote test_preds.csv (regression). "
                f"y_pred_found={res.y_pred_found} | backup={res.backup_path}"
            )
        else:
            res = overwrite_test_preds_classification(
                test_csv_path=test_csv,
                chemprop_test_preds_path=chemprop_test_preds_path,
                label_col=label_col,
                smiles_col=compound_col,
                score_col=None,
                backup=overwrite_backup,
            )
            print(
                f"[INFO] Overwrote test_preds.csv (classification). "
                f"backup={res.backup_path}"
            )

        # -----------------------------
        # 3) Compute ensemble mean/std metrics from checkpoints and overwrite:
        #    - foldX/test_scores.csv
        #    - foldX/fold_0/test_scores.json
        # -----------------------------
        scores = compute_fold_ensemble_scores(
            fold_out_dir=fold_out_dir,
            test_csv_path=test_csv,
            dataset_type=dataset_type,      # "regression" | "classification"
            smiles_col=compound_col,
            label_col=label_col,
            ensemble_size=ensemble_size,
            gpu=gpu,
            no_cuda=no_cuda,
            batch_size=batch_size_eval,
            num_workers=num_workers_eval,
            cls_threshold=0.5,
        )

        scores_csv_path = write_fold_test_scores_csv(
            fold_out_dir=fold_out_dir,
            scores=scores,
            output_name="test_scores.csv",
        )

        scores_json_path = write_fold0_test_scores_json(
            fold_out_dir=fold_out_dir,
            dataset_name=dataset_name,
            scores=scores,
            output_name="test_scores.json",
        )

        print(f"[INFO] Overwrote fold test_scores.csv: {scores_csv_path}")
        print(f"[INFO] Overwrote fold_0 test_scores.json: {scores_json_path}")

    # -----------------------------
    # Save metric.csv at dataset root output directory (fold-wise mean + across-fold mean/std)
    # -----------------------------
    if not dry_run:
        dataset_out_dir = output_dir / model_type / dataset_name
        metric_csv_path = save_metric_csv(
            dataset_out_dir=dataset_out_dir,
            fold_num=fold_num,
            dataset_type=dataset_type,
            output_name="metric.csv",
        )
        print(f"[INFO] Saved metric summary: {metric_csv_path}")



def parse_args() -> argparse.Namespace:
    """
    Args:
        None

    Returns:
        argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Train Chemprop on explicit folds and overwrite test_preds.csv into a standardized format."
    )

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fold_num", type=int, required=True)

    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, choices=["classification", "regression"], default=None)

    parser.add_argument("--compound_col", type=str, required=True)
    parser.add_argument("--label_col", type=str, required=True)

    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--model_type", type=str, default="chemprop_rdkit")

    # Runtime/metric controls
    parser.add_argument("--gpu", type=int, default=None, help="GPU index for Chemprop (--gpu).")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU for Chemprop (--no_cuda).")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for Chemprop (--batch_size).")
    parser.add_argument("--metric", type=str, default=None, help="Primary metric for Chemprop (--metric).")
    parser.add_argument(
        "--extra_metrics",
        type=str,
        nargs="*",
        default=None,
        help="Extra metrics for Chemprop (--extra_metrics ...). Example: --extra_metrics r2",
    )

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    # Overwrite behavior
    parser.add_argument(
        "--overwrite_backup",
        action="store_true",
        help="If set, keep original Chemprop test_preds.csv as test_preds.csv.chemprop.bak before overwriting.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Args:
        None

    Returns:
        None
    """
    args = parse_args()

    train_dataset_dir(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        fold_num=args.fold_num,
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        compound_col=args.compound_col,
        label_col=args.label_col,
        ensemble_size=args.ensemble_size,
        model_type=args.model_type,
        gpu=args.gpu,
        no_cuda=args.no_cuda,
        batch_size=args.batch_size,
        metric=args.metric,
        extra_metrics=args.extra_metrics,
        quiet=args.quiet,
        save_preds=args.save_preds,
        dry_run=args.dry_run,
        overwrite_backup=args.overwrite_backup,
    )


if __name__ == "__main__":
    main()
