"""
saviPGC / PGCMap — paper-oriented methods code (standalone)

This file is a *presentation-ready* organization of the core method code used in:
  1) Sex-aware latent-space calibration (SexHead + temperature scaling),
  2) Reference-guided query mapping (SCANVI query "surgery"),
  3) Sex-compatibility gating + confidence-based Unknown routing,
  4) Optional cross-species ortholog projection and compact summaries (UMAP / Sankey).

Web-service / frontend-backend integration code (FastAPI routing, CORS, HTTP endpoints)
is intentionally omitted here.
"""


# pipeline.py
from __future__ import annotations

import os
import re
import json
from typing import Tuple, Dict, Optional, Sequence, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import scanpy as sc
from anndata import AnnData
import anndata as ad

import scipy.sparse as sp
from scipy.sparse import csr_matrix
import hashlib


# ============================================================
# 0) Cross-species ortholog mapping (query -> human reference)
# ============================================================

def _strip_ensembl_version(x: Any) -> Any:
    if isinstance(x, str):
        return x.split(".")[0]
    return x


def read_biomart_ortholog_table(path: str) -> pd.DataFrame:
    """
    Read a BioMart export table (csv/tsv). This function is tolerant to separator differences.
    """
    tried: list[str] = []
    df: pd.DataFrame | None = None
    for sep in [",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 5:
                return df
        except Exception as e:
            tried.append(f"{sep}: {e}")
    # fallback: let pandas auto-detect
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        return df
    except Exception as e:
        tried.append(f"auto: {e}")
        raise ValueError(f"Failed to read ortholog table: {path}. Tried: {tried}")


def build_one2one_ortholog_maps(
    biomart_path: str,
    target_label: str,
) -> Dict[str, Dict[str, str]]:
    """
    Build 1:1 ortholog mapping dictionaries from a BioMart export table.

    Expected columns (case-sensitive as BioMart):
      - "Gene stable ID", "Gene name" (human)
      - "<Target> gene stable ID", "<Target> gene name"
      - "<Target> homology type"
      - "<Target> orthology confidence [0 low, 1 high]"
      - "%id. target <Target> gene identical to query gene"
      - "%id. query gene identical to target <Target> gene"

    Returns dict with keys:
      - target_eid_to_hsym
      - target_sym_to_hsym
      - target_eid_to_heid
      - target_sym_to_heid
    """
    df = read_biomart_ortholog_table(biomart_path).copy()

    # normalize column names
    human_eid_col = "Gene stable ID"
    human_sym_col = "Gene name"

    target_eid_col = f"{target_label} gene stable ID"
    target_sym_col = f"{target_label} gene name"
    orth_col = f"{target_label} homology type"
    conf_col = f"{target_label} orthology confidence [0 low, 1 high]"
    pid_h2t = f"%id. target {target_label} gene identical to query gene"
    pid_t2h = f"%id. query gene identical to target {target_label} gene"

    missing = [c for c in [human_eid_col, human_sym_col, target_eid_col, orth_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Ortholog table missing columns: {missing}. "
            f"Please export BioMart with human + {target_label} ortholog fields."
        )

    df = df.rename(columns={
        human_eid_col: "human_eid",
        human_sym_col: "human_symbol",
        target_eid_col: "target_eid",
        target_sym_col: "target_symbol",
        orth_col: "orthology_type",
        conf_col: "conf",
        pid_h2t: "pid_h2t",
        pid_t2h: "pid_t2h",
    })

    df["human_eid"] = df["human_eid"].map(_strip_ensembl_version)
    df["target_eid"] = df["target_eid"].map(_strip_ensembl_version)

    one2one = df[df["orthology_type"].astype(str).str.contains("ortholog_one2one", case=False, na=False)].copy()
    if one2one.empty:
        raise ValueError("No 'ortholog_one2one' rows found in ortholog table.")

    def pick_best(g: pd.DataFrame) -> pd.DataFrame:
        gg = g.copy()
        gg["conf"] = pd.to_numeric(gg.get("conf", 0.0), errors="coerce").fillna(0.0)
        pid = np.nanmax(gg[["pid_h2t", "pid_t2h"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).values, axis=1) if (
            "pid_h2t" in gg.columns and "pid_t2h" in gg.columns
        ) else np.zeros(len(gg), dtype=float)
        gg["pid"] = pid
        gg = gg.sort_values(["conf", "pid"], ascending=[False, False])
        return gg.iloc[[0]]

    one2one = one2one.groupby("human_eid", group_keys=False).apply(pick_best)
    one2one = one2one.groupby("target_eid", group_keys=False).apply(pick_best)

    # mappings
    target_eid_to_hsym = one2one.set_index("target_eid")["human_symbol"].dropna().astype(str).to_dict()
    target_eid_to_heid = one2one.set_index("target_eid")["human_eid"].dropna().astype(str).to_dict()

    if "target_symbol" in one2one.columns:
        target_sym_to_hsym = one2one.dropna(subset=["target_symbol"]).set_index("target_symbol")["human_symbol"].astype(str).to_dict()
        target_sym_to_heid = one2one.dropna(subset=["target_symbol"]).set_index("target_symbol")["human_eid"].astype(str).to_dict()
    else:
        target_sym_to_hsym = {}
        target_sym_to_heid = {}

    return {
        "target_eid_to_hsym": target_eid_to_hsym,
        "target_sym_to_hsym": target_sym_to_hsym,
        "target_eid_to_heid": target_eid_to_heid,
        "target_sym_to_heid": target_sym_to_heid,
    }


def ortholog_project_query_to_ref_fast(
    adata_query: AnnData,
    adata_ref: AnnData,
    target_label: str,
    biomart_path: str,
    use_ref_hvg: bool = True,
    layer_name: str = "X_bridge",
) -> AnnData:
    """
    Project a non-human query (Goat/Pig/Macaque/custom) into the human reference gene space.

    - Uses BioMart 1:1 orthologs (best-by-confidence and pid).
    - Aggregates many-to-one mapping by summing counts.
    - Output var_names follow ref HVG order (or full ref var_names if use_ref_hvg=False).
    """
    maps = build_one2one_ortholog_maps(biomart_path, target_label=target_label)

    ref_uses_ensg = pd.Index(adata_ref.var_names).astype(str).str.match(r"^ENSG\d+").mean() > 0.5
    if use_ref_hvg and "highly_variable" in adata_ref.var:
        ref_genes = adata_ref.var_names[adata_ref.var["highly_variable"].astype(bool)]
    else:
        ref_genes = adata_ref.var_names
    ref_genes = pd.Index(ref_genes).astype(str)
    m = int(len(ref_genes))
    ref_pos = pd.Series(np.arange(m, dtype="int64"), index=ref_genes)

    vn = pd.Index(adata_query.var_names).astype(str)
    query_is_ens = vn.str.match(r"^ENS[A-Z]*G\d+").mean() > 0.5

    name_col = next((c for c in ["gene_name", "gene", "symbol", "genes", "Gene", "GeneName"] if c in adata_query.var.columns), None)

    if query_is_ens:
        query_keys = vn.str.replace(r"\.\d+$", "", regex=True)
        mapper = maps["target_eid_to_heid"] if ref_uses_ensg else maps["target_eid_to_hsym"]
    else:
        if name_col is not None:
            query_keys = adata_query.var[name_col].astype(str)
        else:
            query_keys = vn
        mapper = maps["target_sym_to_heid"] if ref_uses_ensg else maps["target_sym_to_hsym"]

    mapped_names = pd.Series(query_keys).map(mapper).to_numpy(object)
    target_idx = pd.Series(mapped_names).map(ref_pos).to_numpy("float64")
    keep = ~np.isnan(target_idx)
    if int(keep.sum()) == 0:
        raise ValueError("No query genes can be mapped to the reference gene set (ortholog mapping coverage = 0).")

    cols = target_idx[keep].astype("int64")
    X = adata_query.X[:, keep]
    if not sp.issparse(X):
        X = csr_matrix(X)

    n_cells, n_cols = X.shape
    rows = np.arange(n_cols, dtype="int64")
    data = np.ones(n_cols, dtype=X.dtype)
    S = csr_matrix((data, (rows, cols)), shape=(n_cols, m))

    X_out = X @ S

    n_sources = np.bincount(cols, minlength=m)
    mapped_ref_idx = np.unique(cols)

    var_df = pd.DataFrame(index=ref_genes)
    var_df["ortholog_mapped"] = False
    var_df.iloc[mapped_ref_idx, var_df.columns.get_loc("ortholog_mapped")] = True
    var_df["ref_index"] = np.arange(m, dtype="int64")
    var_df["n_query_sources"] = n_sources

    adata_out = AnnData(
        X=X_out,
        obs=adata_query.obs.copy(),
        var=var_df,
    )
    adata_out.layers[layer_name] = X_out
    # scvi-tools models often expect raw counts in adata.layers["counts"]
    # If query has been ortholog-projected, the projected matrix is the best available proxy.
    adata_out.layers["counts"] = X_out
    adata_out.uns["ortholog_coverage"] = float(mapped_ref_idx.size) / float(m)
    adata_out.uns["ortholog_target_label"] = str(target_label)
    return adata_out


# ============================================================
# 1) Gene map + 名称统一 + HVG 对齐
# ============================================================

def read_gene_map(path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    读取映射表，返回:
      - id2sym: ENSG -> symbol
      - sym2id: symbol -> ENSG

    自动尝试分隔符：tab / 逗号 / 空白，并鲁棒识别列名。
    """
    tried = []
    df = None
    for sep in ["\t", ",", r"\s+"]:
        try:
            tmp = pd.read_csv(path, sep=sep, engine="python")
            tried.append((sep, tmp.shape))
            if tmp.shape[1] >= 2:
                df = tmp
                break
        except Exception:
            continue

    if df is None or df.shape[1] < 2:
        raise ValueError(f"无法读取 {path}，尝试的分隔与形状：{tried}")

    def norm(s):
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    cols = list(df.columns)
    ncols = {c: norm(c) for c in cols}

    id_candidates, sym_candidates = [], []
    for c, nc in ncols.items():
        if ("gene" in nc) and (("stable" in nc) or ("id" in nc) or ("ensembl" in nc) or ("ensg" in nc)):
            id_candidates.append(c)
        if ("genename" in nc) or ("symbol" in nc) or (("gene" in nc) and ("name" in nc)):
            sym_candidates.append(c)

    if not id_candidates:
        for c, nc in ncols.items():
            if ("ensembl" in nc) or ("ensgid" in nc) or nc == "ensg":
                id_candidates.append(c)
                break

    if not sym_candidates:
        for c, nc in ncols.items():
            if nc in {"symbol", "hgncsymbol"}:
                sym_candidates.append(c)
                break

    if not id_candidates or not sym_candidates:
        print("⚠️ 未找到标准列名，使用前两列作为 (id, symbol)。实际列名：", cols[:2])
        id_col, sym_col = cols[0], cols[1]
    else:
        id_col, sym_col = id_candidates[0], sym_candidates[0]

    def strip_ver(x):
        x = str(x)
        if x.startswith("ENSG"):
            return x.split(".")[0]
        return x

    df[id_col] = df[id_col].map(strip_ver)
    df[sym_col] = df[sym_col].astype(str)

    df = df.dropna(subset=[id_col, sym_col])
    df = df.drop_duplicates([id_col]).drop_duplicates([sym_col])

    id2sym = dict(zip(df[id_col].values, df[sym_col].values))
    sym2id = dict(zip(df[sym_col].values, df[id_col].values))
    return id2sym, sym2id


def is_ensg(s: str) -> bool:
    return isinstance(s, str) and bool(re.match(r"^ENSG\d+", s))


def adata_to_symbols(adata: AnnData, id2sym: Dict[str, str]) -> AnnData:
    """
    把 adata 的 var_names 统一到 gene symbol：
      - 若大多数是 ENSG -> 映射到 symbol
      - 映射不到保留原名
      - 对重复 symbol，保留列总表达量最高的那个
    """
    names = np.array(adata.var_names.astype(str))
    adata_is_ensg = (np.count_nonzero([is_ensg(x) for x in names]) > len(names) / 2)

    if adata_is_ensg:
        mapped = [id2sym.get(x.split(".")[0], None) for x in names]
    else:
        mapped = list(names)

    mapped_fixed = [
        m if (m is not None and m != "" and m != "nan") else n
        for m, n in zip(mapped, names)
    ]

    X = adata.X
    if sp.issparse(X):
        col_sum = np.asarray(X.sum(axis=0)).ravel()
    else:
        col_sum = np.asarray(X.sum(axis=0)).ravel()

    best_idx: Dict[str, int] = {}
    for i, sym in enumerate(mapped_fixed):
        if (sym not in best_idx) or (col_sum[i] > col_sum[best_idx[sym]]):
            best_idx[sym] = i

    keep = np.array(sorted(best_idx.values()))
    ad_sym = adata[:, keep].copy()
    ad_sym.var_names = pd.Index([mapped_fixed[i] for i in keep], name="gene_symbol")
    ad_sym.var_names_make_unique()
    return ad_sym


def align_to_ref_hvg_with_zeros(
    adata_sym: AnnData,
    hvg_ref: Sequence[str],
    id2sym: Dict[str, str],
    use_layer: Optional[str] = None,
    dtype=np.float32,
) -> AnnData:
    """
    将对象按 ref 的 HVG 对齐，缺失基因补0，并保持 ref HVG 的顺序。
    """
    hvg_sym = []
    for g in hvg_ref:
        if is_ensg(g):
            g2 = id2sym.get(g.split(".")[0], None)
            hvg_sym.append(g2 if g2 else g)
        else:
            hvg_sym.append(g)
    hvg_sym = [g for g in hvg_sym if g is not None]

    have = [g for g in hvg_sym if g in adata_sym.var_names]
    missing = [g for g in hvg_sym if g not in adata_sym.var_names]

    A_have = adata_sym[:, have].copy()

    if len(missing) > 0:
        n_cells = A_have.n_obs
        X0 = csr_matrix((n_cells, len(missing)), dtype=dtype)
        ad_missing = AnnData(
            X=X0,
            obs=A_have.obs.copy(),
            var=pd.DataFrame(index=pd.Index(missing, name="gene_symbol"))
        )

        if use_layer is not None:
            if adata_sym.layers is not None and use_layer in adata_sym.layers.keys():
                A_have.layers[use_layer] = adata_sym[:, have].layers[use_layer].copy()
            ad_missing.layers[use_layer] = csr_matrix((n_cells, len(missing)), dtype=dtype)

        A_all = ad.concat([A_have, ad_missing], axis=1, join="outer", merge="unique")
    else:
        A_all = A_have

    A_all = A_all[:, hvg_sym].copy()
    A_all.var["highly_variable"] = A_all.var_names.isin(hvg_sym)
    return A_all


def prepare_query_adata_for_ref(
    adata: AnnData,
    ref: AnnData,
    map_path: str,
    use_layer: Optional[str] = None,
    dtype=np.float32,
) -> Tuple[AnnData, Dict[str, str], Dict[str, str]]:
    """
    一步完成：
      1) 读取映射表
      2) query 统一 gene symbol
      3) 按 ref HVG 对齐并补 0
    """
    id2sym, sym2id = read_gene_map(map_path)

    if "highly_variable" in ref.var.columns:
        hvg_ref = ref.var_names[ref.var["highly_variable"]].tolist()
    else:
        hvg_ref = ref.var_names.tolist()

    adata_sym = adata_to_symbols(adata, id2sym)
    adata_aligned = align_to_ref_hvg_with_zeros(
        adata_sym, hvg_ref, id2sym, use_layer=use_layer, dtype=dtype
    )
    return adata_aligned, id2sym, sym2id


# ============================================================
# 2) Sex head + sex-aware gating + Unknown
# ============================================================

female_only_default = {"RA-responsive", "Oogenesis", "Meiotic prophase"}
male_only_default = {
    "sperm", "Round S'tids", "Early primary S'cytes",
    "Elongated S'tids", "SSCs", "Differentiating S'gonia",
    "Late primary S'cytes", "Mitotic arrest"
}


class SexHead(nn.Module):
    def __init__(self, z_dim, n_classes, dropout=0.2, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),    nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, z):
        return self.net(z)


def load_sex_head(save_dir: str, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(save_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    z_dim = meta["z_dim"]
    idx_to_label = {int(k): v for k, v in meta["idx_to_label"].items()}
    n_classes = len(idx_to_label)
    arch = meta.get("arch", {"dropout": 0.2})

    model = SexHead(z_dim, n_classes, **arch).to(device)
    state = torch.load(os.path.join(save_dir, "sex_head.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    T = float(meta["temperature"])
    return model, T, idx_to_label


def softmax_with_T(logits, T):
    return torch.softmax(logits / T, dim=1)


def get_type_cols_from_ref(ref: AnnData, label_key: str):
    return list(ref.obs[label_key].astype("category").cat.categories)


def get_scanvi_proba(model, adata_q: AnnData, type_cols):
    """
    兼容不同版本 scvi-tools：
      - 优先 predict_proba
      - 否则 predict(soft=True)
    """
    try:
        proba = model.predict_proba(adata_q)
        if isinstance(proba, pd.DataFrame):
            return proba
        return pd.DataFrame(proba, index=adata_q.obs_names, columns=type_cols)
    except Exception:
        pr = model.predict(adata_q, soft=True)
        if isinstance(pr, pd.DataFrame):
            return pr
        return pd.DataFrame(pr, index=adata_q.obs_names, columns=type_cols)


def _sex_key(s: str) -> str:
    s = str(s).strip().casefold()
    if s.startswith("f") or s in {"female", "♀"}:
        return "female"
    if s.startswith("m") or s in {"male", "♂"}:
        return "male"
    return s


def sex_specific_gating(
    proba_df: pd.DataFrame,
    sex_pred: pd.Series,
    sex_conf: pd.Series,
    male_only: set,
    female_only: set,
    thr: float = 0.80,
    soft_eps: float = 1e-2,
) -> pd.DataFrame:
    """
    sex-aware gating：对“对方性别特异类型”做软/硬屏蔽，然后重归一化。
    """
    proba = proba_df.copy()
    for i in proba.index:
        s = _sex_key(sex_pred.loc[i])
        c = float(sex_conf.loc[i])
        if c < thr:
            continue

        if s == "female":
            forbid = set(male_only)
        elif s == "male":
            forbid = set(female_only)
        else:
            continue

        if not forbid:
            continue

        mask = proba.columns.isin(list(forbid))
        if soft_eps == 0.0:
            proba.loc[i, mask] = 0.0
        else:
            proba.loc[i, mask] *= soft_eps

        rs = proba.loc[i].sum()
        if rs > 0:
            proba.loc[i] /= rs

    return proba


def run_sex_aware_scanvi_surgery(
    adata: AnnData,
    lvae,                 # 已训练好的 SCANVI 模型
    ref: AnnData,
    save_dir: str,
    label_key: str = "cell type",
    surgery_epochs: int = 100,
    early_stopping_kwargs: Optional[dict] = None,
    sex_thr: float = 0.90,
    soft_eps: float = 1e-2,
    male_only: Optional[set] = None,
    female_only: Optional[set] = None,
    force_sample_sex: Optional[str] = None,  # "male"/"female"/None
    p_th: float = 0.85,
    h_th: float = 0.65,
    sex_conf_unknown: float = 0.0,
) -> Tuple[AnnData, Any]:
    """
    1) SCANVI load_query_data + train
    2) sex head 性别预测
    3) sex-aware gating
    4) 低置信度/高熵 -> Unknown

    返回：
      - adata_query_latent (X=latent Z, obs=各种结果)
      - surgery_model
    """
    if early_stopping_kwargs is None:
        early_stopping_kwargs = {
            "early_stopping_monitor": "elbo_train",
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.001,
            "plan_kwargs": {"weight_decay": 0.0},
        }

    male_only = male_only if male_only is not None else male_only_default
    female_only = female_only if female_only is not None else female_only_default

    # scvi-tools 的 SCANVI.load_query_data
    from scvi.model import SCANVI

    surgery_model = SCANVI.load_query_data(
        adata,
        lvae,
        freeze_dropout=True,
    )
    surgery_model.train(max_epochs=surgery_epochs, **early_stopping_kwargs)

    z_q = surgery_model.get_latent_representation(adata)
    adata_query_latent = sc.AnnData(z_q)
    adata_query_latent.obs = adata.obs.loc[adata.obs.index, :].copy()

    # 预测（原始）
    cell_type_preds = surgery_model.predict(adata)
    adata_query_latent.obs["predicted_cell_type"] = pd.Categorical(pd.Series(cell_type_preds, index=adata.obs_names).astype(str))


    # sex head (auto -> run model; specified -> skip model)
    if force_sample_sex is not None:
        sex_pred = pd.Series(str(force_sample_sex), index=adata.obs_names)
        sex_conf = pd.Series(1.0, index=adata.obs_names)
        sex_pred_eff = sex_pred
        sex_conf_eff = sex_conf
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sex_head, T, idx2lbl = load_sex_head(save_dir, device=device)

        with torch.no_grad():
            logits = sex_head(torch.from_numpy(z_q).float().to(device))
            sex_proba = softmax_with_T(logits, T).cpu().numpy()

        sex_cols = [idx2lbl[i] for i in range(sex_proba.shape[1])]
        sex_df = pd.DataFrame(sex_proba, index=adata.obs_names, columns=sex_cols)
        sex_pred = sex_df.idxmax(axis=1)
        sex_conf = sex_df.max(axis=1)

        sex_pred_eff = sex_pred
        sex_conf_eff = sex_conf

    # SCANVI proba + gating

    type_cols = get_type_cols_from_ref(ref, label_key=label_key)
    proba_raw = get_scanvi_proba(surgery_model, adata, type_cols=type_cols)
    proba_gated = sex_specific_gating(
        proba_df=proba_raw,
        sex_pred=sex_pred_eff,
        sex_conf=sex_conf_eff,
        male_only=male_only,
        female_only=female_only,
        thr=sex_thr,
        soft_eps=soft_eps,
    )

    labels_raw = proba_raw.idxmax(axis=1)
    labels_gated = proba_gated.idxmax(axis=1)

    adata_query_latent.obs["sex_pred"] = pd.Categorical(pd.Series(sex_pred_eff, index=adata.obs_names).astype(str))
    adata_query_latent.obs["sex_conf"] = pd.Series(sex_conf_eff, index=adata.obs_names).astype(float)
    adata_query_latent.obs["predicted_cell_type_raw"] = pd.Categorical(labels_raw.astype(str))
    adata_query_latent.obs["predicted_cell_type_gated"] = pd.Categorical(labels_gated.astype(str))

    changed = (labels_raw != labels_gated).sum()
    print(
        f"[sex-aware gating] 改变了 {changed} / {proba_raw.shape[0]} 个预测 "
        f"({changed/proba_raw.shape[0]:.1%})；阈值={sex_thr}, 软系数={soft_eps}"
    )

    # 写入每类概率 + 置信度
    proba = proba_gated.copy()
    proba.index.name = None
    adata_query_latent.obs = adata_query_latent.obs.join(proba.add_prefix("p_"), how="left")

    eps = 1e-12
    n_cls = proba.shape[1]
    maxp = proba.max(axis=1).to_numpy()
    entropy = (-(proba * np.log(proba + eps))).sum(axis=1).to_numpy() / np.log(n_cls)

    adata_query_latent.obs["conf_maxp"] = maxp
    adata_query_latent.obs["conf_entropy"] = entropy

    final = np.where(
        (maxp >= p_th)
        & (entropy <= h_th)
        & (adata_query_latent.obs["sex_conf"].astype(float) >= sex_conf_unknown),
        adata_query_latent.obs["predicted_cell_type_gated"].astype(str),
        "Unknown"
    )
    adata_query_latent.obs["predicted_cell_type_final"] = pd.Categorical(final)

    n_total = proba.shape[0]
    n_unknown = int((final == "Unknown").sum())
    print(
        f"[final labeling] Unknown: {n_unknown}/{n_total} "
        f"({n_unknown/n_total:.1%}) | P_TH={p_th}, H_TH={h_th}, SEX_CONF_UNKNOWN={sex_conf_unknown}"
    )

    return adata_query_latent, surgery_model


# ============================================================
# 3) PGC marker filter（表达矩阵上算！）
# ============================================================

def apply_pgc_marker_filter(
    adata: AnnData,
    adata_query_latent: AnnData,
    markers: Optional[Sequence[str]] = None,
    layer_prefer: str = "counts",
    threshold: float = 2.0,
    final_label_key: str = "predicted_cell_type_final",
    pgc_sum_key: str = "pgc_sum",
    unknown_label: str = "Unknown",
) -> Dict[str, Any]:
    """
    在表达矩阵 adata 上计算 marker 总表达 pgc_sum，并同步到 latent，
    然后 pgc_sum < threshold 的细胞强制 final_label_key = Unknown。

    注意：adata 必须是表达矩阵（对齐 HVG 的 query），不要用 latent。
    """
    if markers is None:
        markers = ["PRDM1", "TFAP2C", "KIT", "DPPA3", "DDX4", "DAZL", "PRM2", "PRM1", "ID4"]

    var_upper = pd.Index(adata.var_names.astype(str).str.upper())
    marker_map = {}
    missing = []
    for g in markers:
        g_up = g.upper()
        if g_up in var_upper:
            marker_map[g] = adata.var_names[var_upper.get_loc(g_up)]
        else:
            missing.append(g)

    if missing:
        print("[warn] markers not found in var_names:", missing)
    if not marker_map:
        raise ValueError("None of the markers were found in adata.var_names.")

    cols = [adata.var_names.get_loc(v) for v in marker_map.values()]

    # matrix: prefer counts
    if adata.layers is not None and layer_prefer in adata.layers.keys():
        X = adata.layers[layer_prefer]
        used_layer = layer_prefer
    else:
        X = adata.X
        used_layer = "X"

    M = X[:, cols]
    if sp.issparse(M):
        pgc_sum = np.asarray(M.sum(axis=1)).ravel()
    else:
        pgc_sum = np.asarray(M.sum(axis=1)).ravel()

    adata.obs[pgc_sum_key] = pgc_sum
    print(f"[ok] adata.obs['{pgc_sum_key}'] added. used layer={used_layer}, markers_found={len(cols)}")

    # sync to latent (by index)
    adata_query_latent.obs[pgc_sum_key] = adata.obs[pgc_sum_key].reindex(adata_query_latent.obs_names)

    if final_label_key not in adata_query_latent.obs.columns:
        raise KeyError(f"'{final_label_key}' not found in adata_query_latent.obs")

    mask_false_pgc = (adata_query_latent.obs[pgc_sum_key] < threshold)
    n_forced_unknown = int(mask_false_pgc.sum())
    # 如果是 categorical，确保 Unknown 在 categories 里（否则赋值会报错）
    if pd.api.types.is_categorical_dtype(adata_query_latent.obs[final_label_key]):
        cat = adata_query_latent.obs[final_label_key].cat
        if unknown_label not in cat.categories:
            adata_query_latent.obs[final_label_key] = cat.add_categories([unknown_label])
    adata_query_latent.obs.loc[mask_false_pgc, final_label_key] = unknown_label

    print(f"[pgc_filter] 强制 {n_forced_unknown} 个细胞标签为 '{unknown_label}' (threshold={threshold})")

    return {
        "markers_found": list(marker_map.values()),
        "markers_missing": missing,
        "used_layer": used_layer,
        "n_forced_unknown": n_forced_unknown,
    }


# ============================================================
# 4) Sex-gating 改动分析 -> Sankey nodes/links（给前端）
# ============================================================

def build_final_sankey_data(
    adata_query_latent: AnnData,
    true_label_candidates: Sequence[str] = ("cell_type", "cell type", "CellType"),
    pred_label_col: str = "predicted_cell_type_final",
    unknown_label: str = "Unknown",
    unknown_color: str = "#BDBDBD",
    color_seed: str = "0",
    top_n_links: Optional[int] = None,
) -> Dict[str, Any]:
    """
    构建 sankey：true(cell_type) -> predicted(predicted_cell_type_final)
    - query.obs 中如果找不到 true label 列，则自动创建一列全 Unknown
    - 颜色随机（稳定随机）：同一 label 每次生成的颜色一致
    """

    obs = adata_query_latent.obs

    # 1) 预测列必须存在
    if pred_label_col not in obs.columns:
        raise ValueError(f"Missing `{pred_label_col}` in adata_query_latent.obs")
    pred = obs[pred_label_col].astype('object')
    pred = pred.where(pd.notna(pred), other=unknown_label).astype(str)
    pred = pred.replace({'nan': unknown_label, 'None': unknown_label})

    # 2) 找真实 label 列（cell_type / cell type / ...）
    true_col = None
    for c in true_label_candidates:
        if c in obs.columns:
            true_col = c
            break

    if true_col is None:
        true = pd.Series([unknown_label] * adata_query_latent.n_obs, index=obs.index, name='cell_type')
        true = true.astype('object').fillna(unknown_label).astype(str)
        true = true.replace({'nan': unknown_label, 'None': unknown_label})
        true_col = 'cell_type'
    else:
        true = obs[true_col].astype('object')
        true = true.where(pd.notna(true), other=unknown_label).astype(str)
        true = true.replace({'nan': unknown_label, 'None': unknown_label})

    # 3) 配对表（用于导出 CSV/排查）
    df_pairs = pd.DataFrame(
        {"cell_type": true.values, "predicted_cell_type_final": pred.values},
        index=obs.index
    )

    # 4) 统计 flow（全量，不会空）
    flow_full = (
        df_pairs.groupby(["cell_type", "predicted_cell_type_final"])
        .size()
        .reset_index(name="value")
        .sort_values("value", ascending=False)
    )

    flow_used = flow_full.head(top_n_links).copy() if (top_n_links and top_n_links > 0) else flow_full

    # 5) “changed_cells.csv” 这里改成：真实label != 预测label 的细胞（如果真实列全 Unknown 会几乎全被当 changed，可按需关掉）
    # 你也可以把这行改成 df_pairs.copy() 输出全量
    df_changed = df_pairs[df_pairs["cell_type"].astype(str) != df_pairs["predicted_cell_type_final"].astype(str)].copy()

    # 6) 稳定随机颜色：label -> hex
    def _color_for_label(lab: str) -> str:
        lab = str(lab)
        if lab == unknown_label:
            return unknown_color
        h = hashlib.md5((str(color_seed) + "|" + lab).encode("utf-8")).hexdigest()
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        # 提亮，避免太暗
        r = (r + 128) // 2
        g = (g + 128) // 2
        b = (b + 128) // 2
        return f"#{r:02x}{g:02x}{b:02x}"

    labels_used = pd.unique(df_pairs[['cell_type', 'predicted_cell_type_final']].to_numpy().ravel())
    labels_used = [str(x) for x in labels_used]
    color_map = {lab: _color_for_label(lab) for lab in labels_used}

    # 7) 节点命名：用前缀保证左右两侧同名 label 不冲突
    def src_name(lab: str) -> str:
        return f"cell_type:{lab}"

    def dst_name(lab: str) -> str:
        return f"predicted_final:{lab}"

    src_labels = pd.unique(df_pairs["cell_type"].astype(str))
    dst_labels = pd.unique(df_pairs["predicted_cell_type_final"].astype(str))

    nodes = []
    for lab in src_labels:
        nodes.append({
            "name": src_name(lab),
            "label": str(lab),
            "stage": "cell_type",
            "color": color_map.get(str(lab), "#CCCCCC"),
        })
    for lab in dst_labels:
        nodes.append({
            "name": dst_name(lab),
            "label": str(lab),
            "stage": "predicted",
            "color": color_map.get(str(lab), "#CCCCCC"),
        })

    links = []
    for _, r in flow_used.iterrows():
        s = str(r["cell_type"])
        t = str(r["predicted_cell_type_final"])
        links.append({
            "source": src_name(s),
            "target": dst_name(t),
            "value": int(r["value"]),
        })

    sankey = {
        "nodes": nodes,
        "links": links,
        "color_map": color_map,
        "meta": {
            "true_col_used": true_col,
            "pred_col_used": pred_label_col,
            "n_total": int(adata_query_latent.n_obs),
            "n_pairs": int(flow_full.shape[0]),
        }
    }

    return {
        "summary": sankey["meta"],
        "df_changed": df_changed,
        "flow_full": flow_full,
        "sankey": sankey,
    }


# ============================================================
# 5) UMAP payload（给前端 ECharts scatter）
# ============================================================

def make_umap_payload(
    adata_query_latent: AnnData,
    umap_model,
    label_key: str = "predicted_cell_type_final",
    max_points: Optional[int] = None,
    random_state: int = 0,
) -> list[dict]:
    """
    用 umap_model.transform(Z) 得到 (x,y)，并合并 obs 字段导出 JSON list。
    max_points: 若点太多，可下采样用于前端性能。
    """
    Z = np.asarray(adata_query_latent.X)
    xy = umap_model.transform(Z)

    df = adata_query_latent.obs.copy()
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]
    df["label"] = df[label_key].astype(str) if label_key in df.columns else "NA"

    keep_cols = []
    for c in [
        "x", "y", "label",
        "sex_pred", "sex_conf",
        "pgc_sum", "conf_maxp", "conf_entropy",
        "predicted_cell_type_raw", "predicted_cell_type_gated", "predicted_cell_type_final",
    ]:
        if c in df.columns:
            keep_cols.append(c)

    df = df[keep_cols].copy()
    df = df.reset_index(names="cell_id")

    if max_points is not None and len(df) > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(df), size=max_points, replace=False)
        df = df.iloc[idx].copy()

    # numpy 类型转 python 原生，保证 json.dump 不报错
    payload = df.to_dict(orient="records")
    return payload


# ============================================================
# 6) 上传文件 -> AnnData（先稳妥支持 .h5ad；csv/txt 提供简单兼容）
# ============================================================

def read_uploaded_as_anndata(path: str) -> AnnData:
    """
    - .h5ad: scanpy 直接读
    - .csv/.txt: 假设行为细胞、列为基因（header=gene），第一列可能是 cell_id
    """
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".h5ad":
        return sc.read_h5ad(path)

    if suffix in {".csv", ".txt"}:
        # 自动推断分隔符
        df = pd.read_csv(path, sep=None, engine="python")
        # 若第一列看起来像 cell_id（非数值且唯一），当作 index
        if df.shape[1] >= 2:
            c0 = df.columns[0]
            if df[c0].is_unique and df[c0].astype(str).str.match(r"^[A-Za-z0-9_.:-]+$").mean() > 0.8:
                df = df.set_index(c0)

        X = df.to_numpy()
        adata = AnnData(X=X)
        adata.obs_names = df.index.astype(str)
        adata.var_names = df.columns.astype(str)
        return adata

    raise ValueError(f"Unsupported file type: {suffix}")


# ============================================================
# SexHead training + calibration (as used for the method section)
# ============================================================


import os, json, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


SEX_KEY   = "Gender"
VAL_FRAC  = 0.15
TEST_FRAC = 0.15            # ← 新增：测试集占比
BATCH     = 2048
SAVE_DIR  = "sex_head_ref2"
SEED      = 42
USE_MLP = True
HIDDEN  = 128
DROPOUT = 0.2
LR      = 8e-4
WD      = 2e-4
PATIENCE = 12
EPOCHS   = 120


def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def stratified_split(X: np.ndarray, y: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(y)); tr_idx, va_idx = [], []
    for c in np.unique(y):
        c_idx = idxs[y == c]; rng.shuffle(c_idx)
        n_val = max(1, int(round(len(c_idx) * val_frac)))
        va_idx.extend(c_idx[:n_val].tolist())
        tr_idx.extend(c_idx[n_val:].tolist())
    return np.array(tr_idx), np.array(va_idx)

# —— 新增：三段分割（train / val / test）
def stratified_split_3way(y: np.ndarray, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=SEED):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(y)); tr, va, te = [], [], []
    for c in np.unique(y):
        c_idx = idxs[y == c]; rng.shuffle(c_idx)
        n = len(c_idx)
        n_te = max(1, int(round(n * test_frac)))
        n_va = max(1, int(round(n * val_frac)))
        te.extend(c_idx[:n_te].tolist())
        va.extend(c_idx[n_te:n_te+n_va].tolist())
        tr.extend(c_idx[n_te+n_va:].tolist())
    return np.array(tr), np.array(va), np.array(te)

def to_device(x, device):
    if isinstance(x, (list, tuple)): return [to_device(t, device) for t in x]
    return x.to(device, non_blocking=True)

# ============== model ==============
class SexHead(nn.Module):
    def __init__(self, z_dim, n_classes, dropout=0.2, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),    nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )
    def forward(self, z): return self.net(z)  # logits


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 50) -> float:
    device = logits.device
    T = torch.nn.Parameter(torch.ones([], device=device))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    def loss_fn():
        temp = torch.clamp(T, min=1e-2)
        ll = torch.log_softmax(logits / temp, dim=1)
        return nn.NLLLoss()(ll, targets)
    def closure():
        opt.zero_grad(set_to_none=True); loss = loss_fn(); loss.backward(); return loss
    opt.step(closure)
    return float(torch.clamp(T.detach(), min=1e-2).item())

def softmax_with_T(logits: torch.Tensor, T: float) -> torch.Tensor:
    return torch.softmax(logits / T, dim=1)


@dataclass
class TrainArtifacts:
    state_dict: dict
    idx_to_label: Dict[int, str]
    temperature: float
    z_dim: int
    split: Optional[Dict[str, List[int]]] = None  # ← 新增：保存划分索引（与过滤 NA 后的数组对齐）

def get_z_from_model(model, adata):
    return model.get_latent_representation(adata)

def train_sex_head_on_ref(model, adata_ref, sex_key: str = SEX_KEY,
                          use_mlp: bool = USE_MLP, hidden: int = HIDDEN, dropout: float = DROPOUT,
                          lr: float = LR, wd: float = WD, epochs: int = EPOCHS, patience: int = PATIENCE,
                          batch_size: int = BATCH, seed: int = SEED,
                          val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC,
                          device: Optional[str] = None) -> TrainArtifacts:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")


    z_all = get_z_from_model(model, adata_ref)  # [N, z_dim]
    y_raw_all = adata_ref.obs[sex_key].astype(str).values
    valid_mask = ~pd.isna(y_raw_all)

    z = z_all[valid_mask]
    y_raw = y_raw_all[valid_mask]

    classes = np.unique(y_raw).tolist()
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    y = np.array([label_to_idx[c] for c in y_raw], dtype=np.int64)


    tr_idx, va_idx, te_idx = stratified_split_3way(y, val_frac=val_frac, test_frac=test_frac, seed=seed)
    z_tr, y_tr = z[tr_idx], y[tr_idx]
    z_va, y_va = z[va_idx], y[va_idx]


    binc = np.bincount(y_tr, minlength=len(classes)).astype(np.float32)
    w = binc.sum() / np.maximum(binc, 1.0)

    ds_tr = TensorDataset(torch.from_numpy(z_tr).float(), torch.from_numpy(y_tr))
    ds_va = TensorDataset(torch.from_numpy(z_va).float(), torch.from_numpy(y_va))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, pin_memory=True)

    z_dim = z.shape[1]; n_classes = len(classes)
    net = SexHead(z_dim, n_classes, use_mlp=use_mlp, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))

    best_loss, best_state, wait = math.inf, None, 0
    for ep in range(epochs):
        net.train(); total = 0.0
        for xb, yb in dl_tr:
            xb, yb = to_device((xb, yb), device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = net(xb); loss = criterion(logits, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item() * xb.size(0)
        tr_loss = total / len(ds_tr)

        net.eval(); vtotal = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = to_device((xb, yb), device)
                logits = net(xb); loss = criterion(logits, yb)
                vtotal += loss.item() * xb.size(0)
        va_loss = vtotal / len(ds_va)

        if va_loss < best_loss - 1e-6:
            best_loss, best_state, wait = va_loss, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break

    if best_state is not None: net.load_state_dict(best_state)


    net.eval()
    with torch.no_grad():
        xb = torch.from_numpy(z_va).float().to(device)
        yb = torch.from_numpy(y_va).to(device)
        logits_va = net(xb)
    T = fit_temperature(logits_va, yb)

    return TrainArtifacts(
        state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
        idx_to_label=idx_to_label,
        temperature=T,
        z_dim=z_dim,
        split={
            "valid_mask": valid_mask.astype(bool).tolist(),  # 与 adata_ref.obs_names 对齐
            "train_idx": tr_idx.tolist(),
            "val_idx":   va_idx.tolist(),
            "test_idx":  te_idx.tolist(),
            "classes":   classes,
            "seed":      seed,
            "val_frac":  val_frac,
            "test_frac": test_frac,
        }
    )


def save_sex_head(art: TrainArtifacts, save_dir: str = SAVE_DIR, use_mlp: bool = USE_MLP,
                  hidden: int = HIDDEN, dropout: float = DROPOUT):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(art.state_dict, os.path.join(save_dir, "sex_head.pt"))
    meta = {
        "idx_to_label": art.idx_to_label,
        "label_to_idx": {v: int(k) for k, v in art.idx_to_label.items()},
        "temperature": art.temperature,
        "z_dim": art.z_dim,
        "arch": {"use_mlp": use_mlp, "hidden": hidden, "dropout": dropout},
        "version": 2,                # ← 升级版本号
        "split": art.split,          # ← 保存划分
    }
    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[✓] sex head saved to {save_dir} (with split)")

def load_sex_head(save_dir: str = SAVE_DIR, device: Optional[str] = None) -> Tuple[nn.Module, float, Dict[int,str]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(save_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    z_dim = meta["z_dim"]
    idx_to_label = {int(k): v for k, v in meta["idx_to_label"].items()}
    n_classes = len(idx_to_label)
    arch = meta.get("arch", {"use_mlp": False, "hidden": 64, "dropout": 0.0})
    model = SexHead(z_dim, n_classes, **arch).to(device)
    state = torch.load(os.path.join(save_dir, "sex_head.pt"), map_location=device)
    model.load_state_dict(state); model.eval()
    T = float(meta["temperature"])
    return model, T, idx_to_label


def train_and_save_sex_head_from_ref(model, adata_ref, save_dir: str = SAVE_DIR,
                                     val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC):
    art = train_sex_head_on_ref(model, adata_ref, sex_key=SEX_KEY,
                                val_frac=val_frac, test_frac=test_frac)
    save_sex_head(art, save_dir=save_dir)
    return save_dir

