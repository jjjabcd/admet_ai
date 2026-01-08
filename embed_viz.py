from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
from chemprop.utils import load_checkpoint
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SplitEmbeddings:
    """
    Args:
        embeddings (np.ndarray): Embedding matrix, shape (N, D).
        labels (np.ndarray): Label array, shape (N,).
        smiles (List[str]): SMILES list, length N.

    Returns:
        None
    """
    embeddings: np.ndarray
    labels: np.ndarray
    smiles: List[str]


def _normalize_smiles_cell(x: str) -> str:
    """
    Args:
        x (str): Raw SMILES cell.

    Returns:
        str: Normalized SMILES string.
    """
    s = str(x).strip()

    if s.startswith("['") and s.endswith("']"):
        s = s[2:-2].strip()
    if s.startswith('["') and s.endswith('"]'):
        s = s[2:-2].strip()

    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1].strip()

    return s


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


def _read_csv_split(csv_path: Path, smiles_col: str, label_col: str) -> Tuple[List[str], np.ndarray]:
    """
    Args:
        csv_path (Path): Path to split CSV.
        smiles_col (str): SMILES column name.
        label_col (str): Label column name.

    Returns:
        Tuple[List[str], np.ndarray]:
            - smiles list
            - labels array (float), NaN allowed
    """
    df = pd.read_csv(csv_path)

    if smiles_col not in df.columns:
        raise ValueError(f"{smiles_col} not found in {csv_path}. Columns={list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"{label_col} not found in {csv_path}. Columns={list(df.columns)}")

    smiles = df[smiles_col].astype(str).map(_normalize_smiles_cell).tolist()
    labels = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=float)

    return smiles, labels


def _read_fold_splits(fold_dir: Path, smiles_col: str, label_col: str) -> Tuple[Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray]]:
    """
    Args:
        fold_dir (Path): e.g., DATA_DIR/fold1
        smiles_col (str): SMILES column name
        label_col (str): Label column name (e.g., Ssel)

    Returns:
        Tuple:
            - ((train+val smiles, train+val labels), (test smiles, test labels))
    """
    train_csv = fold_dir / "selectivity_train.csv"
    val_csv = fold_dir / "selectivity_val.csv"
    test_csv = fold_dir / "selectivity_test.csv"

    for p in (train_csv, val_csv, test_csv):
        if not p.exists():
            raise FileNotFoundError(p)

    tr_smiles, tr_y = _read_csv_split(train_csv, smiles_col=smiles_col, label_col=label_col)
    va_smiles, va_y = _read_csv_split(val_csv, smiles_col=smiles_col, label_col=label_col)
    te_smiles, te_y = _read_csv_split(test_csv, smiles_col=smiles_col, label_col=label_col)

    tv_smiles = tr_smiles + va_smiles
    tv_y = np.concatenate([tr_y, va_y], axis=0)

    return (tv_smiles, tv_y), (te_smiles, te_y)


def _build_loader(smiles: List[str], batch_size: int, num_workers: int) -> MoleculeDataLoader:
    """
    Args:
        smiles (List[str]): SMILES list.
        batch_size (int): Batch size.
        num_workers (int): DataLoader workers.

    Returns:
        MoleculeDataLoader: Chemprop MoleculeDataLoader.
    """
    # Chemprop expects each MoleculeDatapoint.smiles as List[str] (supports multi-molecule datapoints).
    dataset = MoleculeDataset([MoleculeDatapoint(smiles=[s]) for s in smiles])

    return MoleculeDataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        shuffle=False,
    )


def _get_single_checkpoint(model_dir: Path, fold_idx: int, model_index: int) -> Path:
    """
    Args:
        model_dir (Path): MODEL_DIR like results/chemprop_rdkit/Kd
        fold_idx (int): Fold index (1-based).
        model_index (int): Which model_i to use under fold_0 (default 0).

    Returns:
        Path: Checkpoint path: fold{idx}/fold_0/model_{model_index}/model.pt
    """
    p = model_dir / f"fold{fold_idx}" / "fold_0" / f"model_{int(model_index)}" / "model.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing checkpoint: {p}")
    return p


def _find_first_linear_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Args:
        model (torch.nn.Module): Loaded model.

    Returns:
        torch.nn.Module: First Linear layer.

    Raises:
        AttributeError: If no Linear layer exists.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            return m
    raise AttributeError("No torch.nn.Linear layer found in model. Cannot hook FFN input.")


def _find_last_linear_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Args:
        model (torch.nn.Module): Loaded model.

    Returns:
        torch.nn.Module: Last Linear layer.

    Raises:
        AttributeError: If no Linear layer exists.
    """
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    if last is None:
        raise AttributeError("No torch.nn.Linear layer found in model.")
    return last


def _select_ffn_input_linear(model: torch.nn.Module) -> torch.nn.Module:
    """
    Select a Linear layer whose input approximates "FFN input embedding".

    Preferred:
        - If model has 'ffn', use the first Linear inside it.
    Fallback:
        - Use first Linear while avoiding last Linear if possible.

    Args:
        model (torch.nn.Module): Loaded model.

    Returns:
        torch.nn.Module: Linear layer to hook.
    """
    if hasattr(model, "ffn"):
        ffn = getattr(model, "ffn")
        if isinstance(ffn, torch.nn.Sequential):
            for layer in ffn:
                if isinstance(layer, torch.nn.Linear):
                    return layer
        for _, layer in ffn.named_modules():
            if isinstance(layer, torch.nn.Linear):
                return layer

    first = _find_first_linear_layer(model)
    last = _find_last_linear_layer(model)
    if first is last:
        return first
    return first


def _extract_ffn_input_from_encoder(
    model: torch.nn.Module,
    loader: MoleculeDataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Args:
        model (torch.nn.Module): Chemprop MoleculeModel.
        loader (MoleculeDataLoader): Data loader.
        device (torch.device): Device.

    Returns:
        np.ndarray: Encoder outputs (FFN inputs), shape (N, D).
    """
    from rdkit import Chem

    model.eval()
    model.to(device)

    outs: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            batch_mols: List[List[Chem.Mol]] = []
            for dp in batch:
                smi = dp.smiles[0]
                mol = Chem.MolFromSmiles(str(smi))
                batch_mols.append([mol])

            # encoder output == FFN input in chemprop 1.6.1
            enc = model.encoder(batch_mols)
            outs.append(enc.detach().cpu())

    z = torch.cat(outs, dim=0).numpy().astype(float)
    return z

def _extract_embeddings_single_model(
    checkpoint_path: Path,
    smiles: List[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    """
    Args:
        checkpoint_path (Path): model.pt path.
        smiles (List[str]): SMILES list.
        batch_size (int): Batch size.
        num_workers (int): DataLoader workers.
        device (torch.device): Device.

    Returns:
        np.ndarray: Embeddings (N, D).
    """
    loader = _build_loader(smiles=smiles, batch_size=batch_size, num_workers=num_workers)

    model = load_checkpoint(path=str(checkpoint_path), device=device).eval()
    linear = _select_ffn_input_linear(model)

    z = _extract_ffn_input_from_encoder(model=model, loader=loader, device=device)
    return z.astype(float)


def save_split_embeddings_npy(out_path: Path, split: SplitEmbeddings, label_col: str) -> None:
    """
    Args:
        out_path (Path): Output npy path.
        split (SplitEmbeddings): Split embeddings container.
        label_col (str): Label key name (e.g., Ssel).

    Returns:
        None
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embeddings": split.embeddings.astype(float),
        label_col: split.labels.astype(float),
        "smiles": np.array(split.smiles, dtype=object),
    }
    np.save(out_path, payload, allow_pickle=True)
    print(f"[INFO] Saved embeddings to {out_path}")


def doPCA(embeddings: np.ndarray, labels: np.ndarray, random_state: int, label_col: str) -> pd.DataFrame:
    """
    Args:
        embeddings (np.ndarray): (N, D)
        labels (np.ndarray): (N,)
        random_state (int): Random seed
        label_col (str): Label column name

    Returns:
        pd.DataFrame: ['PC1','PC2',label_col]
    """
    pca = PCA(n_components=2, random_state=int(random_state))
    xy = pca.fit_transform(embeddings).astype(float)
    return pd.DataFrame({"PC1": xy[:, 0], "PC2": xy[:, 1], label_col: labels.astype(float)})


def doUMAP(
    embeddings: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
    label_col: str,
) -> pd.DataFrame:
    """
    Args:
        embeddings (np.ndarray): (N, D)
        labels (np.ndarray): (N,)
        random_state (int): Random seed
        n_neighbors (int): UMAP n_neighbors
        min_dist (float): UMAP min_dist
        label_col (str): Label column name

    Returns:
        pd.DataFrame: ['UMAP1','UMAP2',label_col]
    """
    import umap  # type: ignore

    reducer = umap.UMAP(
        n_components=2,
        random_state=int(random_state),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
    )
    xy = reducer.fit_transform(embeddings).astype(float)
    return pd.DataFrame({"UMAP1": xy[:, 0], "UMAP2": xy[:, 1], label_col: labels.astype(float)})


def scatterplot_pca(df: pd.DataFrame, filepath: Path, label_col: str, title: str) -> None:
    """
    Args:
        df (pd.DataFrame): ['PC1','PC2',label_col]
        filepath (Path): Output png path
        label_col (str): Color column name
        title (str): Title

    Returns:
        None
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scatter = ax.scatter(df["PC1"], df["PC2"], c=df[label_col], cmap="rainbow", alpha=0.6, s=20)

    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC2", fontsize=15)
    ax.set_title(title, fontsize=16)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(label_col, fontsize=12)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot to {filepath}")


def scatterplot_umap(df: pd.DataFrame, filepath: Path, label_col: str, title: str) -> None:
    """
    Args:
        df (pd.DataFrame): ['UMAP1','UMAP2',label_col]
        filepath (Path): Output png path
        label_col (str): Color column name
        title (str): Title

    Returns:
        None
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scatter = ax.scatter(df["UMAP1"], df["UMAP2"], c=df[label_col], cmap="rainbow", alpha=0.6, s=20)

    ax.set_xlabel("UMAP1", fontsize=15)
    ax.set_ylabel("UMAP2", fontsize=15)
    ax.set_title(title, fontsize=16)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(label_col, fontsize=12)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot to {filepath}")


def parse_args() -> argparse.Namespace:
    """
    Args:
        None

    Returns:
        argparse.Namespace
    """
    p = argparse.ArgumentParser(description="Single-model (per fold) FFN-input embedding extraction + PCA/UMAP (png+npy only).")

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--model_dir", type=str, required=True)

    p.add_argument("--fold_num", type=int, required=True)
    p.add_argument("--smiles_col", type=str, required=True)
    p.add_argument("--label_col", type=str, required=True)

    p.add_argument("--model_index", type=int, default=0, help="Use fold{idx}/fold_0/model_{model_index}/model.pt")

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)

    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--no_cuda", action="store_true")

    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--do_umap", action="store_true")
    p.add_argument("--umap_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)

    return p.parse_args()


def main() -> None:
    """
    Args:
        None

    Returns:
        None
    """
    args = parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()

    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    if int(args.fold_num) < 1:
        raise ValueError("--fold_num must be >= 1")

    device = _infer_device(gpu=args.gpu, no_cuda=bool(args.no_cuda))

    meta: Dict[str, object] = {
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "fold_num": int(args.fold_num),
        "smiles_col": str(args.smiles_col),
        "label_col": str(args.label_col),
        "model_index": int(args.model_index),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "device": str(device),
        "seed": int(args.seed),
        "do_umap": bool(args.do_umap),
        "umap_neighbors": int(args.umap_neighbors),
        "umap_min_dist": float(args.umap_min_dist),
    }

    for fold_idx in range(1, int(args.fold_num) + 1):
        print("\n" + "=" * 80)
        print(f"[Fold {fold_idx}] Processing...")
        print("=" * 80)

        fold_data_dir = data_dir / f"fold{fold_idx}"
        (tv_smiles, tv_y), (te_smiles, te_y) = _read_fold_splits(
            fold_dir=fold_data_dir,
            smiles_col=str(args.smiles_col),
            label_col=str(args.label_col),
        )

        # Output paths
        embeddings_dir = model_dir / f"fold{fold_idx}" / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        with open(embeddings_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        tr_val_path = embeddings_dir / "tr_val_embeddings.npy"
        te_path = embeddings_dir / "te_embeddings.npy"

        # 1) extract embedding (single fixed model)
        ckpt = _get_single_checkpoint(model_dir=model_dir, fold_idx=fold_idx, model_index=int(args.model_index))
        print(f"[INFO] Using checkpoint: {ckpt}")

        print(f"[INFO] Extracting embeddings and saving to {embeddings_dir}")
        emb_train_val = _extract_embeddings_single_model(
            checkpoint_path=ckpt,
            smiles=tv_smiles,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
        )
        emb_test = _extract_embeddings_single_model(
            checkpoint_path=ckpt,
            smiles=te_smiles,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
        )

        save_split_embeddings_npy(
            tr_val_path,
            SplitEmbeddings(embeddings=emb_train_val, labels=tv_y, smiles=tv_smiles),
            label_col=str(args.label_col),
        )
        save_split_embeddings_npy(
            te_path,
            SplitEmbeddings(embeddings=emb_test, labels=te_y, smiles=te_smiles),
            label_col=str(args.label_col),
        )

        print(f"Train+Val embeddings shape: {emb_train_val.shape}")
        print(f"Test embeddings shape: {emb_test.shape}")

        # 2) PCA visualization
        print("[INFO] Performing PCA...")
        df_train_val_pca = doPCA(emb_train_val, tv_y, random_state=int(args.seed), label_col=str(args.label_col))
        df_test_pca = doPCA(emb_test, te_y, random_state=int(args.seed), label_col=str(args.label_col))

        # 3) save png
        pca_dir = embeddings_dir / "PCA"
        pca_dir.mkdir(parents=True, exist_ok=True)

        scatterplot_pca(
            df_train_val_pca,
            pca_dir / "train_val_embeddings.png",
            label_col=str(args.label_col),
            title=f"Train+Val Embeddings Distribution (Fold {fold_idx})",
        )
        scatterplot_pca(
            df_test_pca,
            pca_dir / "test_embeddings.png",
            label_col=str(args.label_col),
            title=f"Test Embeddings Distribution (Fold {fold_idx})",
        )

        # 4) UMAP visualization
        if bool(args.do_umap):
            print("[INFO] Performing UMAP...")
            df_train_val_umap = doUMAP(
                emb_train_val,
                tv_y,
                random_state=int(args.seed),
                n_neighbors=int(args.umap_neighbors),
                min_dist=float(args.umap_min_dist),
                label_col=str(args.label_col),
            )
            df_test_umap = doUMAP(
                emb_test,
                te_y,
                random_state=int(args.seed),
                n_neighbors=int(args.umap_neighbors),
                min_dist=float(args.umap_min_dist),
                label_col=str(args.label_col),
            )

            # 5) save png
            umap_dir = embeddings_dir / "UMAP"
            umap_dir.mkdir(parents=True, exist_ok=True)

            scatterplot_umap(
                df_train_val_umap,
                umap_dir / "train_val_embeddings.png",
                label_col=str(args.label_col),
                title=f"Train+Val Embeddings Distribution (Fold {fold_idx})",
            )
            scatterplot_umap(
                df_test_umap,
                umap_dir / "test_embeddings.png",
                label_col=str(args.label_col),
                title=f"Test Embeddings Distribution (Fold {fold_idx})",
            )

        print(f"\n[INFO] Fold {fold_idx} Statistics:")
        print(
            f"  Train+Val embeddings: shape={emb_train_val.shape}, mean={float(np.nanmean(emb_train_val)):.4f}, std={float(np.nanstd(emb_train_val)):.4f}"
        )
        print(
            f"  Test embeddings: shape={emb_test.shape}, mean={float(np.nanmean(emb_test)):.4f}, std={float(np.nanstd(emb_test)):.4f}"
        )

        if np.isfinite(tv_y).any():
            print(
                f"  Train+Val {args.label_col}: min={float(np.nanmin(tv_y)):.4f}, max={float(np.nanmax(tv_y)):.4f}, mean={float(np.nanmean(tv_y)):.4f}"
            )
        if np.isfinite(te_y).any():
            print(
                f"  Test {args.label_col}: min={float(np.nanmin(te_y)):.4f}, max={float(np.nanmax(te_y)):.4f}, mean={float(np.nanmean(te_y)):.4f}"
            )

        print(f"[Fold {fold_idx}] Completed!")

    print("\n" + "=" * 80)
    print("[INFO] All folds processed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
