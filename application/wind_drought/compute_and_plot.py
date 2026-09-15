#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig1 (v4) release: 风电干旱指标（场站网格）年度聚合与折线图（WDF/WDD 分开画）

指标定义（详见同目录 README.md）：
- WDF：累计小时数 CF < 0.1
- WDD：事件为 CF < Pxx 的连续段；若两段之间间歇 ≤6h 且间歇内 CF ≤ P40，则合并为一个事件
- WDD_mean：当年 active 且事件数>0 的场站网格上 (Dur/Evt) 的均值；阴影为 ±1 std

两种用法：
- 从头计算：--compute（读逐年 wcf NetCDF + 装机表，写入 cache/）
- 仅绘图：--plot（读 cache/ 中已有年度 npz，不碰 NetCDF）

本发布包仅包含代码。--plot 需要用户提供年度缓存；--compute 需要原始数据。
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class Paths:
    data_dir: str
    meta_wind_csv: str
    out_dir: str
    cache_dir: str


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _find_legacy_file(basename: str, legacy_dirs: List[str]) -> Optional[str]:
    for d in legacy_dirs:
        if not d:
            continue
        p = os.path.join(d, basename)
        if os.path.exists(p):
            return p
    return None


def _load_points_any_capacity(meta_csv: str, years: List[int]) -> pd.DataFrame:
    """
    选择“风电场站网格点并集”：任一年 capacity_{y}>0 的点位。
    """
    try:
        df = pd.read_csv(meta_csv, encoding="utf-8")
    except Exception:
        df = pd.read_csv(meta_csv, encoding="gbk")
    for col in ("lat", "lon"):
        if col not in df.columns:
            raise ValueError(f"`{col}` not found in {meta_csv}")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    cols = [f"capacity_{int(y)}" for y in years if f"capacity_{int(y)}" in df.columns]
    if cols:
        cap = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        mask = cap.max(axis=1).values > 0
        return df.loc[mask].copy().reset_index(drop=True)
    if "capacity_2024" in df.columns:
        mask = pd.to_numeric(df["capacity_2024"], errors="coerce").fillna(0.0).values > 0
        return df.loc[mask].copy().reset_index(drop=True)
    raise ValueError(f"No capacity columns found in {meta_csv}")


def _filter_points_extent(df: pd.DataFrame, extent: Tuple[float, float, float, float]) -> pd.DataFrame:
    lon0, lon1, lat0, lat1 = extent
    m = (df["lon"].values >= lon0) & (df["lon"].values <= lon1) & (df["lat"].values >= lat0) & (df["lat"].values <= lat1)
    return df.loc[m].copy().reset_index(drop=True)


def _map_points_to_grid(ds: xr.Dataset, pts: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    lat_vals = ds["lat"].values
    lon_vals = ds["lon"].values
    lat_idx = np.searchsorted(lat_vals, pts["lat"].values)
    lon_idx = np.searchsorted(lon_vals, pts["lon"].values)
    lat_idx = np.clip(lat_idx, 0, len(lat_vals) - 1).astype(np.int32)
    lon_idx = np.clip(lon_idx, 0, len(lon_vals) - 1).astype(np.int32)
    return lat_idx, lon_idx


def _cache_wcf_points_year(
    nc_path: str,
    *,
    out_npy: str,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    chunk_hours: int = 72,
) -> str:
    """
    读取 wcf(time,lat,lon) -> 抽取到 (time,P) 并缓存为 .npy。
    统一输出为 8760 小时：闰年(8784)剔除 2/29 的 24 小时（index 1416..1439）。
    """
    if os.path.exists(out_npy):
        return out_npy
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)

    import netCDF4 as nc  # local import
    from numpy.lib.format import open_memmap

    out_tmp = out_npy + ".tmp"
    if os.path.exists(out_tmp):
        try:
            os.remove(out_tmp)
        except OSError:
            pass

    with nc.Dataset(nc_path) as ds:
        if "wcf" not in ds.variables:
            raise KeyError(f"wcf not in {nc_path}")
        v = ds.variables["wcf"]  # (time, lat, lon)
        try:
            v.set_auto_maskandscale(False)
        except Exception:
            pass

        n_time = int(v.shape[0])
        P = int(lat_idx.size)
        if n_time == 8784:
            drop0, drop1 = 1416, 1440
            out_time = 8760
        else:
            drop0, drop1 = -1, -1
            out_time = n_time
        if out_time != 8760:
            raise ValueError(f"Unexpected time length in {nc_path}: {n_time} (expected 8760 or 8784)")

        mm = open_memmap(out_tmp, mode="w+", dtype=np.float32, shape=(out_time, P))
        step = int(chunk_hours)
        for t0 in range(0, n_time, step):
            t1 = min(n_time, t0 + step)
            block = np.asarray(v[t0:t1, :, :], dtype=np.float32)
            if n_time == 8784:
                src_idx = np.arange(t0, t1, dtype=np.int32)
                keep = (src_idx < drop0) | (src_idx >= drop1)
                if not np.any(keep):
                    continue
                dst_idx = src_idx[keep].copy()
                dst_idx[dst_idx >= drop1] -= 24
                mm[dst_idx, :] = block[keep, :, :][:, lat_idx, lon_idx]
            else:
                mm[t0:t1, :] = block[:, lat_idx, lon_idx]
        mm.flush()
        del mm

    os.replace(out_tmp, out_npy)
    return out_npy


def _merge_wdd_events(drought: np.ndarray, low40: np.ndarray, gap_hours: int) -> np.ndarray:
    """
    若两段 drought(CF<Pxx) 之间间隙长度 <= gap_hours 且间隙内所有小时满足 low40(CF<=P40) 且非 drought，
    则将间隙桥接为 drought（合并为单事件）。
    """
    T = int(drought.shape[0])
    if gap_hours <= 0 or T <= 2:
        return drought
    D = drought.copy()
    nP = int(D.shape[1])
    for k in range(1, int(gap_hours) + 1):
        n = T - k - 1
        if n <= 0:
            continue
        pre = D[:n, :]
        post = D[(k + 1) :, :]
        bridge = pre & post
        if not np.any(bridge):
            continue
        gap_ok = np.ones((n, nP), dtype=bool)
        for o in range(1, k + 1):
            gap_ok &= (~D[o : o + n, :]) & low40[o : o + n, :]
        bridge &= gap_ok
        if not np.any(bridge):
            continue
        for o in range(1, k + 1):
            D[o : o + n, :] |= bridge
    return D


def _compute_thresholds_multi(
    *,
    cache_dir: str,
    wcf_cache: Dict[int, str],
    baseline_years: List[int],
    P: int,
    points_chunk: int,
    drought_percentiles: List[float],
    merge_upper_percentile: float,
    legacy_cache_dirs: List[str],
) -> Tuple[Dict[float, np.ndarray], np.ndarray, str]:
    """
    计算/读取基准期阈值：
    - drought 阈值：P15/P20/P30（每点一值）
    - merge 阈值：P40（每点一值）
    """
    percs = sorted(set(float(x) for x in drought_percentiles))
    p_merge = float(merge_upper_percentile)
    all_q = percs + [p_merge]
    tag = "_".join([f"p{int(round(q*100))}" for q in all_q])
    fn = f"wind_thresh_cf_{tag}_{baseline_years[0]}_{baseline_years[-1]}_P{P}.npz"
    fn_untagged = f"wind_thresh_cf_{tag}_{baseline_years[0]}_{baseline_years[-1]}.npz"
    out_path = os.path.join(cache_dir, fn)

    def _try_load(path: str) -> Optional[Tuple[Dict[float, np.ndarray], np.ndarray]]:
        if not os.path.exists(path):
            return None
        z = np.load(path)
        thresh: Dict[float, np.ndarray] = {}
        for q in percs:
            k = f"p{int(round(q*100))}"
            arr = z[k].astype(np.float32)
            if arr.shape != (P,):
                return None
            thresh[q] = arr
        k_merge = f"p{int(round(p_merge*100))}"
        arr_merge = z[k_merge].astype(np.float32)
        if arr_merge.shape != (P,):
            return None
        return thresh, arr_merge

    loaded = _try_load(out_path)
    if loaded is None:
        # 旧文件名不含 P；仅当点数一致时复用，避免 v3(P=3641) 误用 P3135 阈值
        for cand in [os.path.join(cache_dir, fn_untagged)] + [
            p for p in (_find_legacy_file(fn, legacy_cache_dirs), _find_legacy_file(fn_untagged, legacy_cache_dirs)) if p
        ]:
            loaded = _try_load(cand)
            if loaded is not None:
                if cand != out_path:
                    _link_or_copy(cand, out_path)
                break

    if loaded is not None:
        thresh, arr_merge = loaded
        return thresh, arr_merge, out_path

    # compute
    years = list(baseline_years)
    nY = len(years)
    nT = 8760
    thresh: Dict[float, np.ndarray] = {q: np.full((P,), np.nan, dtype=np.float32) for q in percs}
    thresh_merge = np.full((P,), np.nan, dtype=np.float32)

    for p0 in range(0, P, int(points_chunk)):
        p1 = min(P, p0 + int(points_chunk))
        buf = np.empty((nY * nT, p1 - p0), dtype=np.float32)
        row = 0
        for y in years:
            wcf = np.load(wcf_cache[int(y)], mmap_mode="r")[:, p0:p1]
            buf[row : row + nT, :] = wcf
            row += nT
        qs = np.quantile(buf, all_q, axis=0).astype(np.float32)  # (len(all_q), chunkP)
        for i, q in enumerate(percs):
            thresh[q][p0:p1] = qs[i, :]
        thresh_merge[p0:p1] = qs[len(percs), :]

    save_dict = {f"p{int(round(q*100))}": thresh[q] for q in percs}
    save_dict[f"p{int(round(p_merge*100))}"] = thresh_merge
    np.savez_compressed(out_path, **save_dict)
    return thresh, thresh_merge, out_path


def compute_metrics_v4(
    paths: Paths,
    *,
    baseline_years: List[int],
    target_years: List[int],
    drought_percentiles: List[float],
    china_only: bool = True,
    extent: Tuple[float, float, float, float] = (70.0, 140.0, 15.0, 55.0),
    points_chunk: int = 400,
    wdf_threshold: float = 0.1,
    wdd_merge_gap_hours: int = 6,
    wdd_merge_upper_percentile: float = 0.4,
    legacy_cache_dirs: Optional[List[str]] = None,
) -> str:
    _ensure_dirs(paths.out_dir, paths.cache_dir)
    legacy_cache_dirs = legacy_cache_dirs or []

    years_all = sorted(set(int(y) for y in (baseline_years + target_years)))
    pts = _load_points_any_capacity(paths.meta_wind_csv, years_all)
    if china_only:
        pts = _filter_points_extent(pts, extent)
    P = int(len(pts))
    if P <= 0:
        raise RuntimeError("No active wind station grids after filtering.")

    sample_nc = os.path.join(paths.data_dir, f"{baseline_years[0]}.nc")
    if not os.path.exists(sample_nc):
        raise FileNotFoundError(sample_nc)
    with xr.open_dataset(sample_nc, decode_times=False) as ds0:
        lat_vals = ds0["lat"].values.astype(np.float32)
        lon_vals = ds0["lon"].values.astype(np.float32)
        n_time = int(ds0.sizes["time"])
        lat_idx, lon_idx = _map_points_to_grid(ds0, pts)
    if n_time not in (8760, 8784):
        raise ValueError(f"Unexpected time length: {n_time}")

    # yearly capacity masks
    cap_masks: Dict[int, np.ndarray] = {}
    for y in years_all:
        col = f"capacity_{int(y)}"
        if col in pts.columns:
            cap = pd.to_numeric(pts[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            cap = np.zeros((P,), dtype=np.float64)
        cap_masks[int(y)] = cap > 0

    # cache wcf
    wcf_cache: Dict[int, str] = {}
    for y in years_all:
        nc_path = os.path.join(paths.data_dir, f"{y}.nc")
        if not os.path.exists(nc_path):
            raise FileNotFoundError(nc_path)
        out_npy = os.path.join(paths.cache_dir, f"wcf_points_{y}_P{P}.npy")
        wcf_cache[int(y)] = out_npy
        if os.path.exists(out_npy):
            print(f"[wcf] reuse {out_npy}")
            continue
        legacy = _find_legacy_file(os.path.basename(out_npy), legacy_cache_dirs)
        if legacy:
            _link_or_copy(legacy, out_npy)
            print(f"[wcf] link {legacy} -> {out_npy}")
            continue
        print(f"[wcf] extract {nc_path} -> {out_npy} (P={P})")
        _cache_wcf_points_year(nc_path, out_npy=out_npy, lat_idx=lat_idx, lon_idx=lon_idx, chunk_hours=72)

    # thresholds: P15/P20/P30 + P40(merge)
    thresh_p, thresh_p40, thresh_path = _compute_thresholds_multi(
        cache_dir=paths.cache_dir,
        wcf_cache=wcf_cache,
        baseline_years=list(baseline_years),
        P=P,
        points_chunk=int(points_chunk),
        drought_percentiles=list(drought_percentiles),
        merge_upper_percentile=float(wdd_merge_upper_percentile),
        legacy_cache_dirs=list(legacy_cache_dirs),
    )

    years_t = list(target_years)
    nYt = len(years_t)
    percs = sorted(set(float(x) for x in drought_percentiles))
    nPerc = len(percs)

    # WDF aggregates
    yearly_active_cnt = np.zeros((nYt,), dtype=np.int64)
    yearly_wdf_sum = np.zeros((nYt,), dtype=np.float64)
    yearly_wdf_mean = np.full((nYt,), np.nan, dtype=np.float64)
    yearly_wdf_std = np.full((nYt,), np.nan, dtype=np.float64)

    # WDD aggregates by percentile (station-mean Dur/Evt)
    yearly_wddmean_mean = np.full((nYt, nPerc), np.nan, dtype=np.float64)
    yearly_wddmean_std = np.full((nYt, nPerc), np.nan, dtype=np.float64)
    yearly_wddmean_valid_cnt = np.zeros((nYt, nPerc), dtype=np.int64)

    for yi, y in enumerate(years_t):
        y_int = int(y)
        active = cap_masks.get(y_int, np.zeros((P,), dtype=bool))
        n_active = int(np.count_nonzero(active))
        yearly_active_cnt[yi] = n_active
        if n_active == 0:
            continue

        wcf = np.load(wcf_cache[y_int], mmap_mode="r")  # (8760,P)

        # --- WDF ---
        m_wdf = (wcf < float(wdf_threshold)) & active[None, :]
        wdf_hours = m_wdf.sum(axis=0, dtype=np.int64).astype(np.float64)  # (P,)
        vals = wdf_hours[active]
        yearly_wdf_sum[yi] = float(vals.sum())
        yearly_wdf_mean[yi] = float(vals.mean())
        yearly_wdf_std[yi] = float(vals.std(ddof=0))

        # --- WDD for each percentile ---
        low40 = wcf <= thresh_p40[None, :]
        for pi, q in enumerate(percs):
            drought = wcf < thresh_p[q][None, :]
            m_wdd = _merge_wdd_events(drought, low40, int(wdd_merge_gap_hours))
            m_wdd &= active[None, :]

            dur = m_wdd.sum(axis=0, dtype=np.int64).astype(np.float64)
            starts = m_wdd[0, :].astype(np.int64) + (m_wdd[1:, :] & (~m_wdd[:-1, :])).sum(axis=0, dtype=np.int64)
            evt = starts.astype(np.float64)

            valid = active & (evt > 0)
            cnt = int(np.count_nonzero(valid))
            yearly_wddmean_valid_cnt[yi, pi] = cnt
            if cnt <= 0:
                continue
            mean_dur = np.divide(dur, evt, out=np.full_like(dur, np.nan), where=(evt > 0))
            vv = mean_dur[valid]
            yearly_wddmean_mean[yi, pi] = float(np.nanmean(vv))
            yearly_wddmean_std[yi, pi] = float(np.nanstd(vv, ddof=0))

    # save yearly cache
    perc_tag = "_".join([f"p{int(round(q*100))}" for q in percs])
    yearly_npz = os.path.join(paths.cache_dir, f"wind_yearly_totals_v4_{years_t[0]}_{years_t[-1]}_{perc_tag}_P{P}.npz")
    np.savez_compressed(
        yearly_npz,
        years=np.array(years_t, dtype=np.int32),
        P=np.array([P], dtype=np.int32),
        yearly_active_cnt=yearly_active_cnt,
        yearly_wdf_sum=yearly_wdf_sum,
        yearly_wdf_mean=yearly_wdf_mean,
        yearly_wdf_std=yearly_wdf_std,
        wdd_percentiles=np.array(percs, dtype=np.float32),
        yearly_wddmean_mean=yearly_wddmean_mean,
        yearly_wddmean_std=yearly_wddmean_std,
        yearly_wddmean_valid_cnt=yearly_wddmean_valid_cnt,
        thresh_path=np.array([thresh_path], dtype=object),
        wdf_threshold=np.array([float(wdf_threshold)], dtype=np.float32),
        wdd_merge_gap_hours=np.array([int(wdd_merge_gap_hours)], dtype=np.int32),
        wdd_merge_upper_percentile=np.array([float(wdd_merge_upper_percentile)], dtype=np.float32),
    )
    return yearly_npz


def _pick_latest_npz(cache_dir: str, prefix: str) -> str:
    cand = []
    for fn in os.listdir(cache_dir):
        if fn.startswith(prefix) and fn.endswith(".npz"):
            cand.append(os.path.join(cache_dir, fn))
    if not cand:
        raise FileNotFoundError(f"No cache found with prefix={prefix}")
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def plot_wdf_mean_std_and_sum(cache_dir: str, out_dir: str, *, years: List[int], P: Optional[int] = None) -> str:
    _ensure_dirs(out_dir)
    if P is None:
        npz_path = _pick_latest_npz(cache_dir, f"wind_yearly_totals_v4_{years[0]}_{years[-1]}")
    else:
        npz_path = _pick_latest_npz(cache_dir, f"wind_yearly_totals_v4_{years[0]}_{years[-1]}_")  # tolerate suffix

    z = np.load(npz_path, allow_pickle=True)
    yrs = z["years"].astype(np.int32)
    wdf_sum = z["yearly_wdf_sum"].astype(np.float64)
    wdf_mean = z["yearly_wdf_mean"].astype(np.float64)
    wdf_std = z["yearly_wdf_std"].astype(np.float64)

    fig = plt.figure(figsize=(9.6, 5.0))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    c_mean = "#1f77b4"
    c_sum = "#2ca02c"

    # left axis: mean ± std
    ax1.plot(yrs, wdf_mean, color=c_mean, lw=2.2, marker="o", ms=4, label=r"WDF$_{mean}$")
    ax1.fill_between(
        yrs,
        wdf_mean - wdf_std,
        wdf_mean + wdf_std,
        color=c_mean,
        alpha=0.18,
        linewidth=0,
        label=r"WDF$_{mean}$ ±1 std",
    )

    # right axis: sum
    ax2.plot(yrs, wdf_sum, color=c_sum, lw=2.2, marker="s", ms=4, label=r"WDF$_{sum}$")

    ax1.set_xlabel("Year", fontsize=18)
    ax1.set_ylabel(r"WDF$_{mean}$ (h a$^{-1}$)", fontsize=18, color=c_mean)
    ax2.set_ylabel(r"WDF$_{sum}$ (h a$^{-1}$)", fontsize=18, color=c_sum)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_xlim(int(yrs.min()) - 0.6, int(yrs.max()) + 0.6)
    ax1.tick_params(axis="x", labelsize=14)
    ax1.tick_params(axis="y", labelsize=14, colors=c_mean)
    ax2.tick_params(axis="y", labelsize=14, colors=c_sum)
    ax1.spines["left"].set_color(c_mean)
    ax2.spines["right"].set_color(c_sum)
    ax2.yaxis.get_offset_text().set_color(c_sum)

    def _fmt_sum(v: float) -> str:
        if not np.isfinite(float(v)):
            return ""
        av = abs(float(v))
        if av >= 1e6:
            return f"{float(v)/1e6:.2f}"
        return f"{float(v):.0f}"

    for x, yv in zip(yrs.tolist(), wdf_mean.tolist()):
        if np.isfinite(float(yv)):
            # 2021 与 WDF_sum 交叉：mean 标在折线下方，其余年份标在点上方
            below = int(x) == 2021
            ax1.annotate(
                f"{float(yv):.1f}",
                (int(x), float(yv)),
                textcoords="offset points",
                xytext=(0, -12) if below else (0, 8),
                ha="center",
                va="top" if below else "bottom",
                fontsize=10,
                color="#1f77b4",
                annotation_clip=False,
            )
    for x, yv in zip(yrs.tolist(), wdf_sum.tolist()):
        if np.isfinite(float(yv)):
            # 2014 靠近底轴、2021 交叉点：sum 标在折线上方；其余标点下方
            above = int(x) in (2014, 2021)
            ax2.annotate(
                _fmt_sum(float(yv)),
                (int(x), float(yv)),
                textcoords="offset points",
                xytext=(0, 8) if above else (0, -12),
                ha="center",
                va="bottom" if above else "top",
                fontsize=10,
                color="#2ca02c",
                annotation_clip=False,
            )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=14)

    out = os.path.join(out_dir, f"Fig1_v4_all_station_WDFmeanStd_WDFsum_{int(yrs.min())}_{int(yrs.max())}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_wdf_mean_std_and_sum_annotated(cache_dir: str, out_dir: str, *, years: List[int], P: Optional[int] = None) -> str:
    """
    与 plot_wdf_mean_std_and_sum 相同，但：
    - 去掉“±1 std”阴影的 legend 项
    - 在图中标注每年的 WDF_mean 与 WDF_sum 数值
    - 输出文件名不同，不覆盖旧图
    """
    _ensure_dirs(out_dir)
    if P is None:
        npz_path = _pick_latest_npz(cache_dir, f"wind_yearly_totals_v4_{years[0]}_{years[-1]}")
    else:
        npz_path = _pick_latest_npz(cache_dir, f"wind_yearly_totals_v4_{years[0]}_{years[-1]}_")  # tolerate suffix

    z = np.load(npz_path, allow_pickle=True)
    yrs = z["years"].astype(np.int32)
    wdf_sum = z["yearly_wdf_sum"].astype(np.float64)
    wdf_mean = z["yearly_wdf_mean"].astype(np.float64)
    wdf_std = z["yearly_wdf_std"].astype(np.float64)

    def _fmt_sum(v: float) -> str:
        if not np.isfinite(float(v)):
            return ""
        av = abs(float(v))
        if av >= 1e6:
            return f"{float(v)/1e6:.2f}"
        return f"{float(v):.0f}"

    fig = plt.figure(figsize=(9.6, 5.0))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    c_mean = "#1f77b4"
    c_sum = "#2ca02c"

    ax1.plot(yrs, wdf_mean, color=c_mean, lw=2.2, marker="o", ms=4, label=r"WDF$_{mean}$")
    ax1.fill_between(
        yrs,
        wdf_mean - wdf_std,
        wdf_mean + wdf_std,
        color=c_mean,
        alpha=0.18,
        linewidth=0,
        label="_nolegend_",
    )
    ax2.plot(yrs, wdf_sum, color=c_sum, lw=2.2, marker="s", ms=4, label=r"WDF$_{sum}$")

    ax1.set_xlabel("Year", fontsize=18)
    ax1.set_ylabel(r"WDF$_{mean}$ (h a$^{-1}$)", fontsize=18, color=c_mean)
    ax2.set_ylabel(r"WDF$_{sum}$ (h a$^{-1}$)", fontsize=18, color=c_sum)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_xlim(int(yrs.min()) - 0.6, int(yrs.max()) + 0.6)
    ax1.tick_params(axis="x", labelsize=14)
    ax1.tick_params(axis="y", labelsize=14, colors=c_mean)
    ax2.tick_params(axis="y", labelsize=14, colors=c_sum)
    ax1.spines["left"].set_color(c_mean)
    ax2.spines["right"].set_color(c_sum)
    ax2.yaxis.get_offset_text().set_color(c_sum)

    # annotate each year's values
    for x, yv in zip(yrs.tolist(), wdf_mean.tolist()):
        if np.isfinite(float(yv)):
            ax1.annotate(
                f"{float(yv):.1f}",
                (int(x), float(yv)),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                va="bottom",
                fontsize=10,
                color="#1f77b4",
                annotation_clip=False,
            )
    for x, yv in zip(yrs.tolist(), wdf_sum.tolist()):
        if np.isfinite(float(yv)):
            # Put 2014 & 2021 labels above the points (user request)
            above = int(x) in (2014, 2021)
            ax2.annotate(
                _fmt_sum(float(yv)),
                (int(x), float(yv)),
                textcoords="offset points",
                xytext=(0, 10) if above else (0, -14),
                ha="center",
                va="bottom" if above else "top",
                fontsize=10,
                color="#2ca02c",
                annotation_clip=False,
            )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=14)

    out = os.path.join(out_dir, f"Fig1_v4_all_station_WDFmeanStd_WDFsum_annot5_{int(yrs.min())}_{int(yrs.max())}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_wdd_mean_std_three(cache_dir: str, out_dir: str, *, years: List[int]) -> str:
    _ensure_dirs(out_dir)
    npz_path = _pick_latest_npz(cache_dir, f"wind_yearly_totals_v4_{years[0]}_{years[-1]}")
    z = np.load(npz_path, allow_pickle=True)
    yrs = z["years"].astype(np.int32)
    percs = z["wdd_percentiles"].astype(np.float32).tolist()
    mean = z["yearly_wddmean_mean"].astype(np.float64)  # (nY, nP)
    std = z["yearly_wddmean_std"].astype(np.float64)  # (nY, nP)

    fig = plt.figure(figsize=(9.6, 5.0))
    ax = plt.gca()

    colors = ["#d62728", "#ff7f0e", "#9467bd"]  # red, orange, purple
    # P30 标上方、P15 标下方、P20 紧贴点下，避免三条线数字叠在一起
    offset_by_p = {15: -12, 20: -2, 30: 10}
    for i, q in enumerate(percs):
        lab = f"P{int(round(float(q) * 100))}"
        c = colors[i % len(colors)]
        y = mean[:, i]
        s = std[:, i]
        ax.plot(yrs, y, color=c, lw=2.2, marker="o", ms=4, label=rf"WDD$_{{mean}}$ ({lab})")
        ax.fill_between(yrs, y - s, y + s, color=c, alpha=0.15, linewidth=0)
        p_int = int(round(float(q) * 100))
        yoff = int(offset_by_p.get(p_int, -2))
        for x, yv in zip(yrs.tolist(), y.tolist()):
            if np.isfinite(float(yv)):
                ax.annotate(
                    f"{float(yv):.2f}",
                    (int(x), float(yv)),
                    textcoords="offset points",
                    xytext=(0, yoff),
                    ha="center",
                    va="bottom" if yoff >= 0 else "top",
                    fontsize=10,
                    color=c,
                    annotation_clip=False,
                )

    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel(r"WDD$_{mean}$ (h)", fontsize=18)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(int(yrs.min()) - 0.6, int(yrs.max()) + 0.6)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=14)

    out = os.path.join(out_dir, f"Fig1_v4_all_station_WDDmeanStd_P15_P20_P30_{int(yrs.min())}_{int(yrs.max())}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_wdd_mean_std_three_annotated(cache_dir: str, out_dir: str, *, years: List[int]) -> str:
    """
    与 plot_wdd_mean_std_three 相同，但在图中标注每个年份、每条线的数值。
    输出文件名不同，不覆盖旧图。
    """
    _ensure_dirs(out_dir)
    npz_path = _pick_latest_npz(cache_dir, f"wind_yearly_totals_v4_{years[0]}_{years[-1]}")
    z = np.load(npz_path, allow_pickle=True)
    yrs = z["years"].astype(np.int32)
    percs = z["wdd_percentiles"].astype(np.float32).tolist()
    mean = z["yearly_wddmean_mean"].astype(np.float64)  # (nY, nP)
    std = z["yearly_wddmean_std"].astype(np.float64)  # (nY, nP)

    fig = plt.figure(figsize=(9.6, 5.0))
    ax = plt.gca()

    colors = ["#d62728", "#ff7f0e", "#9467bd"]  # red, orange, purple
    # User request: P30 labels above, P15 labels below, P20 unchanged (keep previous -2)
    offset_by_p = {15: -14, 20: -2, 30: 10}
    for i, q in enumerate(percs):
        lab = f"P{int(round(float(q) * 100))}"
        c = colors[i % len(colors)]
        y = mean[:, i]
        s = std[:, i]
        ax.plot(yrs, y, color=c, lw=2.2, marker="o", ms=4, label=rf"WDD$_{{mean}}$ ({lab})")
        ax.fill_between(yrs, y - s, y + s, color=c, alpha=0.15, linewidth=0)
        p_int = int(round(float(q) * 100))
        yoff = int(offset_by_p.get(p_int, -2))
        for x, yv in zip(yrs.tolist(), y.tolist()):
            if np.isfinite(float(yv)):
                ax.annotate(
                    f"{float(yv):.2f}",
                    (int(x), float(yv)),
                    textcoords="offset points",
                    xytext=(0, yoff),
                    ha="center",
                    va="bottom" if yoff >= 0 else "top",
                    fontsize=10,
                    color=c,
                    annotation_clip=False,
                )

    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel(r"WDD$_{mean}$ (h)", fontsize=18)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(int(yrs.min()) - 0.6, int(yrs.max()) + 0.6)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=14)

    out = os.path.join(out_dir, f"Fig1_v4_all_station_WDDmeanStd_P15_P20_P30_annot4_{int(yrs.min())}_{int(yrs.max())}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _package_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _nature_energy_root() -> str:
    env = os.environ.get("NATURE_ENERGY_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(_package_dir(), ".."))


def _has_yearly_cache(cache_dir: str, years: List[int]) -> bool:
    try:
        _pick_latest_npz(cache_dir, f"wind_yearly_totals_v4_{years[0]}_{years[-1]}")
        return True
    except FileNotFoundError:
        return False


def main() -> int:
    here = _package_dir()
    root = _nature_energy_root()
    default_data_dir = os.path.join(root, "data", "history_stations_new_nearest_0127")
    default_meta = os.path.join(root, "data", "history_capacity_new_by_province", "grid_cmfd_wind_all_v3.csv")
    default_out = os.path.join(here, "figures")
    default_cache = os.path.join(here, "cache")
    default_legacy = []  # External caches must be explicitly selected.

    ap = argparse.ArgumentParser(description="Compute or plot station-grid WDF/WDD (Fig1 v4).")
    ap.add_argument("--data-dir", default=default_data_dir, help="逐年 NetCDF 目录（变量 wcf）")
    ap.add_argument("--meta-wind-csv", default=default_meta, help="风电装机网格表（capacity_YYYY）")
    ap.add_argument("--out-dir", default=default_out, help="图片输出目录")
    ap.add_argument("--cache-dir", default=default_cache, help="指标缓存目录（年度 npz / 可选小时 npy）")
    ap.add_argument("--legacy-cache-dirs", default=",".join(default_legacy), help="已有 wcf/阈值缓存，逗号分隔")

    ap.add_argument("--compute", action="store_true", help="从 NetCDF 计算 WDF/WDD 并写入 cache")
    ap.add_argument("--plot", action="store_true", help="从年度缓存绘制 WDF 与 WDD 主图")
    ap.add_argument("--plot-wdf", action="store_true", help="输出 WDF_mean±std + WDF_sum（双坐标）")
    ap.add_argument("--plot-wdd", action="store_true", help="输出 WDD_mean±std（P15/P20/P30 三条线）")
    ap.add_argument("--plot-wdf-annot", action="store_true", help="输出带数值标注的 WDF 图（不覆盖主图）")
    ap.add_argument("--plot-wdd-annot", action="store_true", help="输出带数值标注的 WDD 图（不覆盖主图）")
    ap.add_argument("--plot-annot", action="store_true", help="同时输出 WDF/WDD 标注图")

    ap.add_argument("--extent", default="70,140,15,55")
    ap.add_argument("--china-only", action="store_true", default=True)
    ap.add_argument("--no-china-only", dest="china_only", action="store_false")
    ap.add_argument("--points-chunk", type=int, default=400)
    ap.add_argument("--years", default="", help="逗号分隔年份列表；baseline 与 target 使用同一列表（默认=2014-2024锚点年）")
    ap.add_argument("--wdf-threshold", type=float, default=0.1)
    ap.add_argument("--wdd-merge-gap-hours", type=int, default=6)
    ap.add_argument("--wdd-merge-upper-percentile", type=float, default=0.4)
    ap.add_argument("--wdd-percentiles", default="0.15,0.2,0.3", help="WDD drought percentiles, default=0.15,0.2,0.3")
    args = ap.parse_args()

    extent = tuple(float(x) for x in str(args.extent).split(","))
    if len(extent) != 4:
        raise ValueError("--extent must be lon0,lon1,lat0,lat1")

    if str(args.years).strip():
        years = [int(x) for x in str(args.years).split(",") if str(x).strip()]
        years = sorted(set(years))
    else:
        years = [2014, 2017, 2020, 2021, 2022, 2023, 2024]

    drought_percentiles = [float(x) for x in str(args.wdd_percentiles).split(",") if str(x).strip()]
    if not drought_percentiles:
        raise ValueError("--wdd-percentiles is empty")

    if args.plot:
        args.plot_wdf = True
        args.plot_wdd = True
    if args.plot_annot:
        args.plot_wdf_annot = True
        args.plot_wdd_annot = True

    legacy_dirs = [x.strip() for x in str(args.legacy_cache_dirs).split(",") if x.strip()]
    paths = Paths(data_dir=str(args.data_dir), meta_wind_csv=str(args.meta_wind_csv), out_dir=str(args.out_dir), cache_dir=str(args.cache_dir))
    _ensure_dirs(paths.out_dir, paths.cache_dir)

    do_plot = bool(args.plot_wdf or args.plot_wdd or args.plot_wdf_annot or args.plot_wdd_annot)
    if (not args.compute) and (not do_plot):
        if _has_yearly_cache(paths.cache_dir, years):
            args.plot_wdf = True
            args.plot_wdd = True
            do_plot = True
            print("[info] no flags given; yearly cache found -> --plot")
        else:
            ap.print_help()
            return 2

    if args.compute:
        yearly_npz = compute_metrics_v4(
            paths,
            baseline_years=years,
            target_years=years,
            drought_percentiles=drought_percentiles,
            china_only=bool(args.china_only),
            extent=extent,  # type: ignore[arg-type]
            points_chunk=int(args.points_chunk),
            wdf_threshold=float(args.wdf_threshold),
            wdd_merge_gap_hours=int(args.wdd_merge_gap_hours),
            wdd_merge_upper_percentile=float(args.wdd_merge_upper_percentile),
            legacy_cache_dirs=legacy_dirs,
        )
        print(f"[OK] computed yearly cache: {yearly_npz}")

    if args.plot_wdf:
        out = plot_wdf_mean_std_and_sum(paths.cache_dir, paths.out_dir, years=years)
        print(f"[OK] saved: {out}")

    if args.plot_wdd:
        out = plot_wdd_mean_std_three(paths.cache_dir, paths.out_dir, years=years)
        print(f"[OK] saved: {out}")

    if args.plot_wdf_annot:
        out = plot_wdf_mean_std_and_sum_annotated(paths.cache_dir, paths.out_dir, years=years)
        print(f"[OK] saved: {out}")

    if args.plot_wdd_annot:
        out = plot_wdd_mean_std_three_annotated(paths.cache_dir, paths.out_dir, years=years)
        print(f"[OK] saved: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
