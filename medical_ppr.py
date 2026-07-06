# -*- coding: utf-8 -*-
"""
medical_ppr.py

基于新的三节点 my_data.py 重新生成旧 RDCP / HeroFilter 需要的双通道 PPR 文件。

本版重点修改：
1. 不再默认保存到 ./data/ppr，避免和老数据集 PPR 混在一起；
2. 如果不手动指定 --out_dir，会自动生成独立目录：
      <ppr_root>/ppr_<dataset_tag>_3node_alpha<alpha>_top<topk>/
   例如：
      ./data/ppr_mimic3_data3_3node_alpha0.5_top64/
3. 仍然保留 --out_dir，手动指定时完全按你给的目录保存；
4. 输出 manifest.json，记录 dataset_path、alpha、topk、节点数、边数等信息，方便以后审计。

边类型约定：
    0 = Patient-Drug
    1 = Patient-Procedure
    2 = Drug-Patient
    3 = Procedure-Patient

输出文件名与旧 train.py 的 load_dual_ppr 保持一致：
    top_indices_alpha_0.5_medical_drug.npy
    top_values_alpha_0.5_medical_drug.npy
    top_indices_alpha_0.5_medical_proc.npy
    top_values_alpha_0.5_medical_proc.npy
"""

import argparse
import json
import os
import os.path as osp
from pathlib import Path
from datetime import datetime
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import sparse
from tqdm import tqdm

from my_data import load_medical_dataset


def _format_alpha(alpha: float) -> str:
    """保持旧文件名风格：0.5 而不是 0.50。"""
    return str(round(float(alpha), 2))


def _safe_tag(text: str) -> str:
    s = str(text).strip()
    keep = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "dataset"


def _infer_dataset_tag(dataset_path: str) -> str:
    """
    从 dataset_path 推断数据集名。

    支持：
    1. .../mimic3_data3
    2. .../mimic3_data3/active_split
    3. .../mimic3_data3/active_split/graph_raw_tensors.pt
    """
    p = Path(dataset_path).expanduser()

    if p.name == "graph_raw_tensors.pt":
        # .../<dataset>/<split>/graph_raw_tensors.pt
        if p.parent.name in {"active_split"} or p.parent.name.startswith("ratio_"):
            return _safe_tag(p.parent.parent.name)
        return _safe_tag(p.parent.name)

    # .../<dataset>/<split>
    if p.name == "active_split" or p.name.startswith("ratio_"):
        return _safe_tag(p.parent.name)

    # .../<dataset>
    return _safe_tag(p.name)


def _resolve_out_dir(args) -> Path:
    """
    PPR 输出目录优先级：
    1. 用户显式指定 --out_dir：完全使用该路径；
    2. 未指定 --out_dir：自动生成 <ppr_root>/ppr_<tag>_3node_alpha<alpha>_top<topk>。
    """
    if args.out_dir is not None and str(args.out_dir).strip():
        return Path(args.out_dir).expanduser().resolve()

    dataset_tag = _safe_tag(args.tag) if args.tag else _infer_dataset_tag(args.dataset_path)
    alpha_str = _format_alpha(args.alpha)
    folder_name = f"ppr_{dataset_tag}_3node_alpha{alpha_str}_top{int(args.topk)}"

    ppr_root = Path(args.ppr_root).expanduser().resolve()
    return ppr_root / folder_name


def _pagerank_power_scipy(G, personalize, alpha=0.5, max_iter=100, tol=1e-8):
    """
    如果 fast_pagerank 不可用，使用 scipy 实现一个简单 power iteration。
    这里 alpha 与旧脚本保持一致，表示 restart / teleport 概率。
    """
    n = G.shape[0]
    row_sum = np.asarray(G.sum(axis=1)).ravel().astype(np.float64)
    row_sum[row_sum == 0] = 1.0
    P = sparse.diags(1.0 / row_sum).dot(G).tocsr()

    p = personalize.astype(np.float64)
    p_sum = p.sum()
    if p_sum <= 0:
        p[:] = 1.0 / n
    else:
        p = p / p_sum

    x = p.copy()
    for _ in range(max_iter):
        x_new = alpha * p + (1.0 - alpha) * (P.T @ x)
        if np.abs(x_new - x).sum() < tol:
            x = x_new
            break
        x = x_new
    return x


def calculate_pagerank(i, alpha, topk, G, num_nodes, use_fast):
    personalization = np.zeros(num_nodes, dtype=np.float64)
    personalization[i] = 1.0

    if use_fast:
        try:
            from fast_pagerank import pagerank_power
            pr = pagerank_power(G, personalize=personalization, p=alpha)
        except Exception:
            pr = _pagerank_power_scipy(G, personalization, alpha=alpha)
    else:
        pr = _pagerank_power_scipy(G, personalization, alpha=alpha)

    k = min(int(topk), int(num_nodes))
    idx_topk = np.flip(np.argsort(pr)[-k:])
    vals = pr[idx_topk]

    if k < topk:
        pad_n = topk - k
        idx_topk = np.concatenate([idx_topk, np.full(pad_n, i, dtype=np.int64)])
        vals = np.concatenate([vals, np.zeros(pad_n, dtype=np.float64)])

    return i, idx_topk.astype(np.int64), vals.astype(np.float64)


def generate_subgraph_ppr(dataset_name, edge_index_np, num_nodes, alpha, topk, suffix, out_dir, workers, use_fast):
    print(f"\n--- Building [{suffix.upper()}] subgraph PPR ---", flush=True)
    if edge_index_np.shape[1] == 0:
        print(f"Warning: {suffix} edge_index is empty. Output self-neighbor placeholders.", flush=True)
        top_indices = np.tile(np.arange(num_nodes, dtype=np.int64).reshape(-1, 1), (1, topk))
        top_values = np.zeros((num_nodes, topk), dtype=np.float64)
    else:
        G = sparse.csr_matrix(
            (
                np.ones(edge_index_np.shape[1], dtype=np.float64),
                (edge_index_np[0, :], edge_index_np[1, :]),
            ),
            shape=(num_nodes, num_nodes),
        )

        top_indices = np.zeros((num_nodes, topk), dtype=np.int64)
        top_values = np.zeros((num_nodes, topk), dtype=np.float64)

        partial_func = partial(
            calculate_pagerank,
            alpha=alpha,
            topk=topk,
            G=G,
            num_nodes=num_nodes,
            use_fast=use_fast,
        )

        print(f"Calculating PageRank: alpha={alpha}, topk={topk}, nodes={num_nodes}, workers={workers}", flush=True)
        if workers <= 1:
            results = [partial_func(i) for i in tqdm(range(num_nodes), total=num_nodes)]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(tqdm(executor.map(partial_func, range(num_nodes)), total=num_nodes))

        for i, indices, values in results:
            top_indices[i, :] = indices
            top_values[i, :] = values

    os.makedirs(out_dir, exist_ok=True)
    alpha_str = _format_alpha(alpha)
    idx_path = osp.join(out_dir, f"top_indices_alpha_{alpha_str}_{dataset_name}_{suffix}.npy")
    val_path = osp.join(out_dir, f"top_values_alpha_{alpha_str}_{dataset_name}_{suffix}.npy")
    np.save(idx_path, top_indices)
    np.save(val_path, top_values)
    print(f"[{suffix.upper()}] saved:", flush=True)
    print(f"  {idx_path}", flush=True)
    print(f"  {val_path}", flush=True)
    return {
        "suffix": suffix,
        "edge_count": int(edge_index_np.shape[1]),
        "top_indices_path": str(idx_path),
        "top_values_path": str(val_path),
        "top_indices_shape": list(top_indices.shape),
        "top_values_shape": list(top_values.shape),
    }


def _write_manifest(out_dir, args, data, summaries):
    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": str(Path(args.dataset_path).expanduser().resolve()),
        "data_name": args.data,
        "alpha": float(args.alpha),
        "topk": int(args.topk),
        "ppr_root": str(Path(args.ppr_root).expanduser().resolve()),
        "out_dir": str(Path(out_dir).expanduser().resolve()),
        "tag": args.tag or _infer_dataset_tag(args.dataset_path),
        "workers": int(args.workers),
        "use_fast": bool(not args.no_fast),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "node_counts": {
            "patient": int((data.node_type == 0).sum()) if hasattr(data, "node_type") else None,
            "drug": int((data.node_type == 1).sum()) if hasattr(data, "node_type") else None,
            "procedure": int((data.node_type == 2).sum()) if hasattr(data, "node_type") else None,
        },
        "edge_type_counts": {
            str(int(et)): int((data.edge_type == et).sum())
            for et in sorted(data.edge_type.unique().tolist())
        },
        "outputs": summaries,
        "note": "This PPR directory is generated for the 3-node RDCP-compatible graph. Do not mix it with old PPR directories.",
    }
    path = Path(out_dir) / "ppr_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest saved: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate dual-channel PPR for 3-node medical graph")
    #parser.add_argument("--dataset_path", type=str,default=r"D:\tool\data_pipeline_v3\final_datasets\gnn_ready\mimic3_data3")
    parser.add_argument("--dataset_path", type=str, default=r"D:\tool\data_pipeline_v3\final_datasets\mimic3_paper1_5class")
    parser.add_argument("--data", type=str, default="medical", help="用于输出文件名，旧训练脚本一般用 medical")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=64)

    # 新增：自动隔离 PPR 目录。
    parser.add_argument(
        "--ppr_root",
        type=str,
        default=osp.join(".", "data"),
        help="未指定 --out_dir 时，自动在该目录下创建 ppr_<tag>_3node_alpha*_top* 文件夹。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="PPR 输出目录。若指定，则完全使用该目录；若不指定，自动生成独立新目录。",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="PPR 目录名中的数据集标签。默认从 dataset_path 自动推断，例如 mimic3_data3。",
    )

    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no_fast", action="store_true", help="不使用 fast_pagerank，改用 scipy power iteration")
    args = parser.parse_args()

    out_dir = _resolve_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading medical dataset for PPR...", flush=True)
    print(f"dataset_path: {args.dataset_path}", flush=True)
    print(f"PPR out_dir : {out_dir}", flush=True)

    dataset = load_medical_dataset(args.dataset_path, name=args.data, aggregate_lab_to_patient=False)
    data = dataset[0]
    num_nodes = int(data.num_nodes)

    edge_index = data.edge_index
    edge_type = data.edge_type

    print(f"Graph: nodes={num_nodes}, edges={edge_index.size(1)}", flush=True)
    print("Edge type counts:", flush=True)
    for et in sorted(edge_type.unique().tolist()):
        print(f"  type {int(et)}: {int((edge_type == et).sum())}", flush=True)

    summaries = []

    # Drug channel: Patient-Drug + Drug-Patient
    mask_drug = (edge_type == 0) | (edge_type == 2)
    edge_index_drug = edge_index[:, mask_drug].cpu().numpy()
    summaries.append(generate_subgraph_ppr(
        dataset_name=args.data,
        edge_index_np=edge_index_drug,
        num_nodes=num_nodes,
        alpha=args.alpha,
        topk=args.topk,
        suffix="drug",
        out_dir=str(out_dir),
        workers=args.workers,
        use_fast=not args.no_fast,
    ))

    # Procedure channel: Patient-Procedure + Procedure-Patient
    mask_proc = (edge_type == 1) | (edge_type == 3)
    edge_index_proc = edge_index[:, mask_proc].cpu().numpy()
    summaries.append(generate_subgraph_ppr(
        dataset_name=args.data,
        edge_index_np=edge_index_proc,
        num_nodes=num_nodes,
        alpha=args.alpha,
        topk=args.topk,
        suffix="proc",
        out_dir=str(out_dir),
        workers=args.workers,
        use_fast=not args.no_fast,
    ))

    _write_manifest(out_dir, args, data, summaries)

    print("\nAll dual-channel PPR files generated successfully.", flush=True)
    print("Use this ppr_dir for training:", flush=True)
    print(f"  --ppr_dir \"{out_dir}\"", flush=True)


if __name__ == "__main__":
    main()
