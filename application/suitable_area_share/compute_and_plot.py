#!/usr/bin/env python3
"""
Suitable-area share (lonlat 0.1° / optional 10 km window): compute indicators and plot.

Share definition:
- yearly: among grids with capacity_Y > 0, mean of the cell's suitable-area fraction, in percent
- province-year: same, but grouped by province (CSV `province` field)

Area methods:
- lonlat0p1 (release default): 0.1° × 0.1° lon-lat cell, counted on the 1 km Albers mask
- window10km: 10 km × 10 km window in mask CRS

Two usage modes:
- --compute: read capacity CSVs + CERF GeoTIFFs, write stats_*.csv into cache/
- --plot: read cache/ stats and write figures/ (no raster IO)

This code-only package requires external stats for --plot, or raw inputs for --compute.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _read_csv_auto_encoding(path: str) -> pd.DataFrame:
    # v3 为 UTF-8；v2 装机表多为 GBK。先 UTF-8，失败再 GBK，避免把 UTF-8 中文读成乱码。
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def compute_suitable_area_fraction_by_window(
    mask_path: str,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    half_size_m: float = 5000.0,
    suitable_values: Tuple[float, ...] = (0.0,),
    nodata_value: float = -1.0,
) -> np.ndarray:
    """
    面积口径（更物理、更稳健）：
    - 将 (lon,lat) 转到 mask 的投影坐标系
    - 以该点为中心取 10km×10km 窗口（half_size_m=5000m）
    - 在窗口内统计 (value in suitable_values) 的 1km 像元占比（忽略 nodata）

    由于 mask 本身是等面积投影（Albers）且分辨率 1000m，这个“像元计数占比”可直接理解为面积占比。
    """
    if lons.shape != lats.shape:
        raise ValueError("lons/lats shape mismatch")
    try:
        import rasterio  # type: ignore
        from rasterio.windows import from_bounds  # type: ignore
        from pyproj import Transformer  # type: ignore
    except Exception as e:
        raise RuntimeError("需要安装 rasterio + pyproj 才能按面积口径计算 suitable 占比。") from e

    out = np.full((lons.shape[0],), np.nan, dtype=np.float32)
    suit = np.array(list(suitable_values), dtype=np.float32)

    with rasterio.open(mask_path) as ds:
        nod = ds.nodata if ds.nodata is not None else nodata_value
        tfm = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        xs, ys = tfm.transform(lons.astype(float), lats.astype(float))
        for i, (x, y) in enumerate(zip(xs, ys)):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            win = from_bounds(
                float(x - half_size_m),
                float(y - half_size_m),
                float(x + half_size_m),
                float(y + half_size_m),
                transform=ds.transform,
            )
            arr = ds.read(1, window=win, boundless=True, fill_value=nod).astype(np.float32)
            valid = arr != float(nod)
            denom = int(np.count_nonzero(valid))
            if denom <= 0:
                out[i] = np.nan
                continue
            num = int(np.count_nonzero(valid & np.isin(arr, suit)))
            out[i] = float(num / denom)
    return out


def compute_suitable_area_fraction_by_lonlat_cell(
    mask_path: str,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    half_deg: float = 0.05,
    suitable_values: Tuple[float, ...] = (0.0,),
    nodata_value: float = -1.0,
) -> np.ndarray:
    """
    面积口径（0.1°×0.1° 经纬网格版本）：
    - 对每个中心点 (lon,lat)，定义经纬盒子：[lon±half_deg, lat±half_deg]（默认 half_deg=0.05 即 0.1°×0.1°）
    - 将该经纬盒子的四个角投影到 mask CRS（米单位）
    - 在 mask 上读取覆盖该投影多边形的窗口
    - 用像元中心点是否落入投影多边形来筛选像元，并统计 suitable 像元占比

    说明：mask 是等面积投影且像元 1000m，这个“像元计数占比”可解释为面积占比。
    """
    if lons.shape != lats.shape:
        raise ValueError("lons/lats shape mismatch")
    try:
        import rasterio  # type: ignore
        from rasterio.windows import from_bounds  # type: ignore
        from pyproj import Transformer  # type: ignore
        from matplotlib.path import Path  # type: ignore
    except Exception as e:
        raise RuntimeError("需要安装 rasterio + pyproj + matplotlib 才能按 0.1°×0.1° 口径计算面积占比。") from e

    out = np.full((lons.shape[0],), np.nan, dtype=np.float32)
    suit = np.array(list(suitable_values), dtype=np.float32)

    with rasterio.open(mask_path) as ds:
        nod = ds.nodata if ds.nodata is not None else nodata_value
        tfm = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)

        for i, (lon, lat) in enumerate(zip(lons.astype(float), lats.astype(float))):
            if not (np.isfinite(lon) and np.isfinite(lat)):
                continue

            lon0, lon1 = float(lon - half_deg), float(lon + half_deg)
            lat0, lat1 = float(lat - half_deg), float(lat + half_deg)
            # corners (clockwise)
            xs, ys = tfm.transform([lon0, lon1, lon1, lon0], [lat0, lat0, lat1, lat1])
            poly = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
            if not np.isfinite(poly).all():
                continue
            xmin, ymin = float(np.min(poly[:, 0])), float(np.min(poly[:, 1]))
            xmax, ymax = float(np.max(poly[:, 0])), float(np.max(poly[:, 1]))

            win = from_bounds(xmin, ymin, xmax, ymax, transform=ds.transform)
            arr = ds.read(1, window=win, boundless=True, fill_value=nod).astype(np.float32)

            # pixel center coordinates in mask CRS for this window
            wt = ds.window_transform(win)
            h, w = arr.shape
            if h <= 0 or w <= 0:
                continue
            cols = np.arange(w, dtype=np.float32)
            rows = np.arange(h, dtype=np.float32)
            cc, rr = np.meshgrid(cols + 0.5, rows + 0.5)
            xx = wt.c + cc * wt.a + rr * wt.b
            yy = wt.f + cc * wt.d + rr * wt.e

            pts = np.column_stack([xx.ravel(), yy.ravel()])
            inside = Path(poly, closed=True).contains_points(pts).reshape((h, w))

            valid = inside & (arr != float(nod))
            denom = int(np.count_nonzero(valid))
            if denom <= 0:
                out[i] = np.nan
                continue
            num = int(np.count_nonzero(valid & np.isin(arr, suit)))
            out[i] = float(num / denom)

    return out


def _compute_share(
    df: pd.DataFrame,
    years: List[int],
    suitable_area_frac: np.ndarray,
    *,
    province_col: str = "province",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回：
    - yearly: 每个 year 一个值（%），表示“当年有装机的 10km 网格的 suitable 面积占比（平均值）”
    - prov_2024: 2024 年各省一个值（%），表示“该省有装机的 10km 网格的 suitable 面积占比（平均值）”
    """
    if suitable_area_frac.shape[0] != len(df):
        raise ValueError("suitable_area_frac 与 df 行数不一致")

    out_yearly = []
    for y in years:
        cap_col = f"capacity_{int(y)}"
        if cap_col not in df.columns:
            continue
        cap = pd.to_numeric(df[cap_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        has = cap > 0
        denom = int(np.count_nonzero(has))
        if denom > 0:
            v = suitable_area_frac[has]
            share = float(np.nanmean(v) * 100.0) if np.isfinite(v).any() else np.nan
        else:
            share = np.nan
        out_yearly.append({"year": int(y), "n_grids_has_cap": denom, "share_pct": share})
    yearly = pd.DataFrame(out_yearly).sort_values("year").reset_index(drop=True)

    # province (2024)
    cap_col = "capacity_2024"
    if cap_col not in df.columns:
        raise ValueError("输入表缺少 capacity_2024")
    cap24 = pd.to_numeric(df[cap_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    has24 = cap24 > 0
    prov = df[province_col].astype(str).fillna("unknown").to_numpy()
    rows = []
    for p in sorted(set(prov.tolist())):
        if p in ("unknown", "境界线", "", "nan", "None"):
            continue
        idx = prov == p
        denom = int(np.count_nonzero(has24 & idx))
        if denom > 0:
            v = suitable_area_frac[has24 & idx]
            share = float(np.nanmean(v) * 100.0) if np.isfinite(v).any() else np.nan
        else:
            share = np.nan
        rows.append({"province": p, "n_grids_has_cap_2024": denom, "share_pct": share})
    prov_2024 = pd.DataFrame(rows).sort_values("share_pct", ascending=False).reset_index(drop=True)
    return yearly, prov_2024


def _compute_province_share_for_year(
    df: pd.DataFrame,
    year: int,
    suitable_area_frac: np.ndarray,
    *,
    province_col: str = "province",
) -> pd.DataFrame:
    if suitable_area_frac.shape[0] != len(df):
        raise ValueError("suitable_area_frac 与 df 行数不一致")
    cap_col = f"capacity_{int(year)}"
    if cap_col not in df.columns:
        raise ValueError(f"输入表缺少 {cap_col}")
    cap = pd.to_numeric(df[cap_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    has = cap > 0
    prov = df[province_col].astype(str).fillna("unknown").to_numpy()
    rows = []
    for p in sorted(set(prov.tolist())):
        if p in ("unknown", "境界线", "", "nan", "None"):
            continue
        idx = prov == p
        denom = int(np.count_nonzero(has & idx))
        total_cap = float(np.nansum(cap[idx])) if int(np.count_nonzero(idx)) > 0 else 0.0
        if denom > 0:
            v = suitable_area_frac[has & idx]
            share = float(np.nanmean(v) * 100.0) if np.isfinite(v).any() else np.nan
        else:
            share = np.nan
        rows.append({"province": p, "n_grids_has_cap": denom, "total_capacity": total_cap, "share_pct": share})
    return pd.DataFrame(rows).sort_values("share_pct", ascending=False).reset_index(drop=True)


def _set_matplotlib_fonts() -> None:
    # 尽量保证中文省名可显示（系统通常有 Noto CJK）
    import matplotlib
    from matplotlib import font_manager

    matplotlib.rcParams["font.family"] = "sans-serif"
    # conda 环境下 Matplotlib 可能扫不到系统字体；这里显式 addfont
    candidate_font_files = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    added_names: List[str] = []
    for fp in candidate_font_files:
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
                name = font_manager.FontProperties(fname=fp).get_name()
                if name not in added_names:
                    added_names.append(name)
            except Exception:
                pass

    matplotlib.rcParams["font.sans-serif"] = (
        added_names
        + [
            "DejaVu Sans",
            "Arial",
            "Liberation Sans",
        ]
    )
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"
    matplotlib.rcParams["savefig.edgecolor"] = "white"


def _plot_line_anchor_years(
    out_dir: str,
    wind: pd.DataFrame,
    solar: pd.DataFrame,
    *,
    ylabel: str = "Suitable area share within built grids (%)",
    out_name: str = "fig_suitable_area_share_2014_2024_line_anchor_years.png",
    big_fonts: bool = False,
) -> str:
    import matplotlib.pyplot as plt

    _set_matplotlib_fonts()

    fig, ax = plt.subplots(figsize=(10.8, 6.4) if big_fonts else (10.5, 6.2))
    ax.plot(wind["year"], wind["share_pct"], marker="o", lw=2.2, label="Wind")
    ax.plot(solar["year"], solar["share_pct"], marker="o", lw=2.2, label="Solar")
    x0 = int(min(wind["year"].min(), solar["year"].min()))
    x1 = int(max(wind["year"].max(), solar["year"].max()))
    ax.set_xlim(x0 - 0.6, x1 + 0.6)
    ax.set_xlabel("Year", fontsize=24 if big_fonts else None)
    ax.set_ylabel(ylabel, fontsize=24 if big_fonts else None)
    ax.grid(True, alpha=0.25)
    if big_fonts:
        ax.tick_params(axis="both", labelsize=18)
        ax.legend(frameon=False, fontsize=20)
    else:
        ax.legend(frameon=False)

    # annotate values
    for _, r in wind.iterrows():
        if np.isfinite(float(r["share_pct"])):
            ax.annotate(
                f"{float(r['share_pct']):.1f}%",
                (int(r["year"]), float(r["share_pct"])),
                textcoords="offset points",
                # Wind: place label below marker (text extends downward)
                xytext=(0, -10),
                ha="center",
                va="top",
                annotation_clip=False,
                fontsize=18 if big_fonts else None,
            )
    for _, r in solar.iterrows():
        if np.isfinite(float(r["share_pct"])):
            ax.annotate(
                f"{float(r['share_pct']):.1f}%",
                (int(r["year"]), float(r["share_pct"])),
                textcoords="offset points",
                # Solar: place label above marker (text extends upward)
                xytext=(0, 10),
                ha="center",
                va="bottom",
                annotation_clip=False,
                fontsize=18 if big_fonts else None,
            )

    fig.tight_layout()
    p = os.path.join(out_dir, out_name)
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def _plot_yearly_bars_two_panel(
    out_dir: str,
    wind: pd.DataFrame,
    solar: pd.DataFrame,
    *,
    out_name: str,
    big_fonts: bool = False,
) -> str:
    import matplotlib.pyplot as plt

    _set_matplotlib_fonts()

    years = wind["year"].to_numpy(dtype=int)
    wind_pct = wind["share_pct"].to_numpy(dtype=float)
    solar_pct = solar["share_pct"].to_numpy(dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11.5, 9.2) if big_fonts else (11.0, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    # Wind bars
    ax1.bar(years, wind_pct, color="#2b6cb0", alpha=0.92, width=0.65)
    ax1.set_ylabel("Percentage (%)", fontsize=24 if big_fonts else None)
    ax1.grid(True, axis="y", alpha=0.25)

    # Solar bars
    ax2.bar(years, solar_pct, color="#d69e2e", alpha=0.92, width=0.65)
    ax2.set_ylabel("Percentage (%)", fontsize=24 if big_fonts else None)
    ax2.set_xlabel("Year", fontsize=24 if big_fonts else None)
    ax2.grid(True, axis="y", alpha=0.25)

    # ticks + margins
    x0 = int(min(years.min(), solar["year"].min()))
    x1 = int(max(years.max(), solar["year"].max()))
    ax2.set_xlim(x0 - 0.7, x1 + 0.7)
    ax2.set_xticks(sorted(set(years.tolist()) | set(solar["year"].to_numpy(dtype=int).tolist())))

    if big_fonts:
        ax1.tick_params(axis="both", labelsize=18)
        ax2.tick_params(axis="both", labelsize=18)

    # annotate values
    for x, v in zip(years, wind_pct):
        if np.isfinite(v):
            ax1.text(int(x), float(v) + 0.6, f"{float(v):.1f}%", ha="center", va="bottom", fontsize=18 if big_fonts else 12)
    for x, v in zip(solar["year"].to_numpy(dtype=int), solar_pct):
        if np.isfinite(v):
            ax2.text(int(x), float(v) + 0.6, f"{float(v):.1f}%", ha="center", va="bottom", fontsize=18 if big_fonts else 12)

    # panel labels (compact)
    ax1.set_title("Wind", fontsize=22 if big_fonts else 16, loc="left", pad=6)
    ax2.set_title("Solar", fontsize=22 if big_fonts else 16, loc="left", pad=6)

    fig.tight_layout()
    p = os.path.join(out_dir, out_name)
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def _plot_yearly_bars_single(
    out_dir: str,
    df_yearly: pd.DataFrame,
    *,
    color: str,
    title: str,
    out_name: str,
    big_fonts: bool = False,
) -> str:
    import matplotlib.pyplot as plt

    _set_matplotlib_fonts()

    years = df_yearly["year"].to_numpy(dtype=int)
    pct = df_yearly["share_pct"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11.5, 5.8) if big_fonts else (11.0, 5.4))
    ax.bar(years, pct, color=color, alpha=0.92, width=0.65, label=title)

    x0 = int(years.min())
    x1 = int(years.max())
    ax.set_xlim(x0 - 0.7, x1 + 0.7)
    ax.set_xticks(years.tolist())

    ax.set_xlabel("Year", fontsize=24 if big_fonts else None)
    ax.set_ylabel("Percentage (%)", fontsize=24 if big_fonts else None)
    ax.grid(True, axis="y", alpha=0.25)

    if big_fonts:
        ax.tick_params(axis="both", labelsize=18)

    # add vertical padding to avoid annotation clipping
    vmax = float(np.nanmax(pct)) if np.isfinite(pct).any() else 0.0
    # 给 legend 留更充足的顶部空白
    pad = max(10.0, vmax * 0.35)
    ax.set_ylim(0, vmax + pad)

    # annotate values
    for x, v in zip(years, pct):
        if np.isfinite(v):
            ax.text(int(x), float(v) + 0.6, f"{float(v):.1f}%", ha="center", va="bottom", fontsize=18 if big_fonts else 12)

    # legend 放到图框上方，避免遮挡柱子
    ax.legend(
        frameon=False,
        fontsize=20 if big_fonts else None,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    p = os.path.join(out_dir, out_name)
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def _province_cn_to_en() -> Dict[str, str]:
    # 与 CSV 的 province 字段一致（中文全称）
    return {
        "北京市": "Beijing",
        "天津市": "Tianjin",
        "河北省": "Hebei",
        "山西省": "Shanxi",
        "内蒙古自治区": "Inner Mongolia",
        "辽宁省": "Liaoning",
        "吉林省": "Jilin",
        "黑龙江省": "Heilongjiang",
        "上海市": "Shanghai",
        "江苏省": "Jiangsu",
        "浙江省": "Zhejiang",
        "安徽省": "Anhui",
        "福建省": "Fujian",
        "江西省": "Jiangxi",
        "山东省": "Shandong",
        "河南省": "Henan",
        "湖北省": "Hubei",
        "湖南省": "Hunan",
        "广东省": "Guangdong",
        "广西壮族自治区": "Guangxi",
        "海南省": "Hainan",
        "重庆市": "Chongqing",
        "四川省": "Sichuan",
        "贵州省": "Guizhou",
        "云南省": "Yunnan",
        "西藏自治区": "Tibet",
        "陕西省": "Shaanxi",
        "甘肃省": "Gansu",
        "青海省": "Qinghai",
        "宁夏回族自治区": "Ningxia",
        "新疆维吾尔自治区": "Xinjiang",
        "台湾省": "Taiwan",
        "香港特别行政区": "Hong Kong",
        "澳门特别行政区": "Macao",
    }


def _province_cn_to_abbr() -> Dict[str, str]:
    # 首字母缩写（尽量避免歧义）
    return {
        "北京市": "BJ",
        "天津市": "TJ",
        "河北省": "HE",
        "山西省": "SX",
        "内蒙古自治区": "NMG",
        "辽宁省": "LN",
        "吉林省": "JL",
        "黑龙江省": "HLJ",
        "上海市": "SH",
        "江苏省": "JS",
        "浙江省": "ZJ",
        "安徽省": "AH",
        "福建省": "FJ",
        "江西省": "JX",
        "山东省": "SD",
        "河南省": "HEN",
        "湖北省": "HB",
        "湖南省": "HUN",
        "广东省": "GD",
        "广西壮族自治区": "GX",
        "海南省": "HI",
        "重庆市": "CQ",
        "四川省": "SC",
        "贵州省": "GZ",
        "云南省": "YN",
        "西藏自治区": "XZ",
        "陕西省": "SN",
        "甘肃省": "GS",
        "青海省": "QH",
        "宁夏回族自治区": "NX",
        "新疆维吾尔自治区": "XJ",
        "台湾省": "TW",
        "香港特别行政区": "HK",
        "澳门特别行政区": "MO",
    }


def _load_prov_attr_abbr_map(prov_attr_csv: str) -> Dict[str, str]:
    """
    使用 prov_attr.csv 的简称（name -> abbr）。
    文件格式示例：name,code,abbr
    """
    try:
        df = pd.read_csv(prov_attr_csv)
    except Exception:
        try:
            df = pd.read_csv(prov_attr_csv, encoding="gbk")
        except Exception:
            return {}
    if not {"name", "abbr"}.issubset(set(df.columns)):
        return {}
    out: Dict[str, str] = {}
    for _, r in df.iterrows():
        n = str(r["name"]).strip()
        a = str(r["abbr"]).strip()
        if n and a and n.lower() != "nan" and a.lower() != "nan":
            out[n] = a
    return out


def _plot_province_barh_single(
    out_dir: str,
    prov_df: pd.DataFrame,
    *,
    color: str,
    fname: str,
    x_label: str = "Share (%)",
    y_label_mode: str = "en",  # "en" or "abbr"
    filter_nonpositive: bool = False,
    big_fonts: bool = False,
    denom_col: str = "n_grids_has_cap_2024",
    abbr_map: Optional[Dict[str, str]] = None,
    order_by: str = "share_pct",  # "share_pct" or "total_capacity"
) -> str:
    import matplotlib.pyplot as plt

    _set_matplotlib_fonts()

    d = prov_df.copy()
    cn2en = _province_cn_to_en()
    cn2abbr = abbr_map or _province_cn_to_abbr()
    d["province_en"] = d["province"].astype(str).map(cn2en).fillna(d["province"].astype(str))
    d["province_abbr"] = d["province"].astype(str).map(cn2abbr).fillna(d["province_en"])
    d["denom_any"] = pd.to_numeric(d.get(denom_col, np.nan), errors="coerce")
    d["share_plot"] = pd.to_numeric(d["share_pct"], errors="coerce")
    d["total_capacity_plot"] = pd.to_numeric(d.get("total_capacity", np.nan), errors="coerce")
    if filter_nonpositive:
        d = d[np.isfinite(d["share_plot"].to_numpy(dtype=float))].copy()
        d = d[np.isfinite(d["denom_any"].to_numpy(dtype=float)) & (d["denom_any"].to_numpy(dtype=float) > 0)].copy()
        d = d[d["share_plot"].to_numpy(dtype=float) > 0].copy()
    else:
        d.loc[~np.isfinite(d["share_plot"].to_numpy(dtype=float)), "share_plot"] = 0.0

    # 排序：默认为了 barh 视觉上“从上到下是降序”，这里用升序排序（大值会排在最上方）
    if str(order_by) == "total_capacity":
        d = d.sort_values(["total_capacity_plot", "province_en"], ascending=[True, True]).reset_index(drop=True)
    else:
        d = d.sort_values(["share_plot", "province_en"], ascending=[True, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.5, max(7.5, 0.42 * max(1, len(d)))))
    ycol = "province_abbr" if str(y_label_mode) == "abbr" else "province_en"
    ax.barh(d[ycol], d["share_plot"], color=color, alpha=0.92)
    ax.set_xlabel(x_label, fontsize=24 if big_fonts else None)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(0, 100)
    if big_fonts:
        ax.tick_params(axis="both", labelsize=22)

    # annotate values at bar end
    for y, (v, denom) in enumerate(zip(d["share_pct"].to_numpy(dtype=float), d["denom_any"].to_numpy(dtype=float))):
        if (not np.isfinite(denom)) or denom <= 0:
            if not filter_nonpositive:
                ax.text(1.0, y, "NA", va="center", ha="left", fontsize=22 if big_fonts else 10, color="#444444")
            continue
        if not np.isfinite(v):
            if not filter_nonpositive:
                ax.text(1.0, y, "NA", va="center", ha="left", fontsize=22 if big_fonts else 10, color="#444444")
            continue
        ax.text(min(v + 1.0, 99.0), y, f"{v:.1f}%", va="center", ha="left", fontsize=22 if big_fonts else 10)

    fig.tight_layout()
    p = os.path.join(out_dir, fname)
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def _package_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _nature_energy_root() -> str:
    env = os.environ.get("NATURE_ENERGY_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(_package_dir(), ".."))


def _plot_all_from_tables(
    *,
    cache_dir: str,
    out_dir: str,
    tag: str,
    prov_years: List[int],
    abbr_map: Dict[str, str],
) -> List[str]:
    """Read stats_*.csv from cache_dir and write figures to out_dir. Metric logic unchanged."""
    saved: List[str] = []
    wind_yearly_anchor = pd.read_csv(os.path.join(cache_dir, f"stats_wind_suitable_area_share_anchor_years_{tag}.csv"))
    solar_yearly_anchor = pd.read_csv(os.path.join(cache_dir, f"stats_solar_suitable_area_share_anchor_years_{tag}.csv"))

    if tag == "lonlat0p1":
        saved.append(
            _plot_yearly_bars_single(
                out_dir,
                wind_yearly_anchor,
                color="#2b6cb0",
                title="Wind",
                out_name="fig_suitable_area_share_2014_2024_wind_bars_lonlat0p1.png",
                big_fonts=True,
            )
        )
        saved.append(
            _plot_yearly_bars_single(
                out_dir,
                solar_yearly_anchor,
                color="#d69e2e",
                title="Solar",
                out_name="fig_suitable_area_share_2014_2024_solar_bars_lonlat0p1.png",
                big_fonts=True,
            )
        )
        for yy in prov_years:
            wind_prov_Y = pd.read_csv(os.path.join(cache_dir, f"stats_wind_suitable_area_share_{yy}_by_province_{tag}.csv"))
            solar_prov_Y = pd.read_csv(os.path.join(cache_dir, f"stats_solar_suitable_area_share_{yy}_by_province_{tag}.csv"))
            saved.append(
                _plot_province_barh_single(
                    out_dir,
                    wind_prov_Y,
                    color="#2b6cb0",
                    fname=f"fig_suitable_area_share_{yy}_wind_by_province_barh_lonlat0p1.png",
                    x_label="Percentage (%)",
                    y_label_mode="abbr",
                    filter_nonpositive=True,
                    big_fonts=True,
                    denom_col="n_grids_has_cap",
                    abbr_map=abbr_map,
                    order_by="total_capacity",
                )
            )
            saved.append(
                _plot_province_barh_single(
                    out_dir,
                    solar_prov_Y,
                    color="#d69e2e",
                    fname=f"fig_suitable_area_share_{yy}_solar_by_province_barh_lonlat0p1.png",
                    x_label="Percentage (%)",
                    y_label_mode="abbr",
                    filter_nonpositive=True,
                    big_fonts=True,
                    denom_col="n_grids_has_cap",
                    abbr_map=abbr_map,
                    order_by="total_capacity",
                )
            )
    else:
        saved.append(_plot_line_anchor_years(out_dir, wind_yearly_anchor, solar_yearly_anchor))
        for yy in prov_years:
            wind_prov_Y = pd.read_csv(os.path.join(cache_dir, f"stats_wind_suitable_area_share_{yy}_by_province_{tag}.csv"))
            solar_prov_Y = pd.read_csv(os.path.join(cache_dir, f"stats_solar_suitable_area_share_{yy}_by_province_{tag}.csv"))
            saved.append(
                _plot_province_barh_single(
                    out_dir,
                    wind_prov_Y,
                    color="#2b6cb0",
                    fname=f"fig_suitable_area_share_{yy}_wind_by_province_barh.png",
                    denom_col="n_grids_has_cap",
                    order_by="total_capacity",
                )
            )
            saved.append(
                _plot_province_barh_single(
                    out_dir,
                    solar_prov_Y,
                    color="#d69e2e",
                    fname=f"fig_suitable_area_share_{yy}_solar_by_province_barh.png",
                    denom_col="n_grids_has_cap",
                    order_by="total_capacity",
                )
            )
    return saved


def main() -> None:
    here = _package_dir()
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--base-dir", default=_nature_energy_root())
    base_args, _ = base_parser.parse_known_args()
    root = os.path.abspath(base_args.base_dir)
    default_out = os.path.join(here, "figures")
    default_cache = os.path.join(here, "cache")
    default_prov_attr = os.path.join(here, "prov_attr.csv")
    default_wind_csv = os.path.join(root, "data", "history_capacity_new_by_province", "grid_cmfd_wind_all_v3.csv")
    default_solar_csv = os.path.join(root, "data", "history_capacity_new_by_province", "grid_cmfd_solar_all_v2.csv")
    default_wind_mask = os.path.join(
        root, "data", "suitable_position", "China", "gridcerf-china", "compiled",
        "gridcerf_wind_onshore_hubheight100m_strict.tif",
    )
    default_solar_mask = os.path.join(
        root, "data", "suitable_position", "China", "gridcerf-china", "compiled",
        "gridcerf_solar_pv_centralized_strict.tif",
    )

    ap = argparse.ArgumentParser(description="Compute or plot suitable-area share (wind/solar).")
    ap.add_argument("--base-dir", type=str, default=root, help="外部数据根目录，输入路径默认由其推导")
    ap.add_argument("--out-dir", type=str, default=default_out, help="图片输出目录")
    ap.add_argument("--cache-dir", type=str, default=default_cache, help="stats_*.csv 缓存目录")
    ap.add_argument("--wind-csv", type=str, default=default_wind_csv)
    ap.add_argument("--solar-csv", type=str, default=default_solar_csv)
    ap.add_argument("--wind-mask", type=str, default=default_wind_mask)
    ap.add_argument("--solar-mask", type=str, default=default_solar_mask)
    ap.add_argument("--suitable-values", type=str, default="0", help="逗号分隔的像元值（默认 0）")
    ap.add_argument("--min-year", type=int, default=2014)
    ap.add_argument("--max-year", type=int, default=2024)
    ap.add_argument(
        "--area-method",
        type=str,
        default="lonlat0p1",
        choices=["window10km", "lonlat0p1"],
        help="面积口径：lonlat0p1=0.1°×0.1°（发布默认）；window10km=固定10km×10km窗口",
    )
    ap.add_argument("--province-years", type=str, default="2014,2024", help="省份 bar 图统计年份，逗号分隔")
    ap.add_argument("--prov-attr-csv", type=str, default=default_prov_attr, help="省份简称映射表（name,abbr）")
    ap.add_argument("--compute", action="store_true", help="从装机表+适宜性栅格计算 stats 并写入 cache")
    ap.add_argument("--plot", action="store_true", help="从 cache 中 stats_*.csv 出图（不读栅格）")
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="同 --plot（兼容旧脚本）",
    )
    args = ap.parse_args()

    cache_dir = str(args.cache_dir)
    out_dir = str(args.out_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    tag = "lonlat0p1" if str(args.area_method) == "lonlat0p1" else "window10km"
    prov_years = [int(x) for x in str(args.province_years).split(",") if str(x).strip() != ""]
    prov_years = sorted(set(prov_years)) or [2024]
    abbr_map = _load_prov_attr_abbr_map(str(args.prov_attr_csv))

    do_compute = bool(args.compute)
    do_plot = bool(args.plot or args.plot_only)
    yearly_csv = os.path.join(cache_dir, f"stats_wind_suitable_area_share_anchor_years_{tag}.csv")
    if (not do_compute) and (not do_plot):
        if os.path.exists(yearly_csv):
            do_plot = True
            print("[info] no flags given; stats cache found -> --plot")
        else:
            ap.print_help()
            return

    if do_compute:
        suitable_values = tuple(float(x) for x in str(args.suitable_values).split(",") if str(x).strip() != "")
        if not suitable_values:
            suitable_values = (0.0,)

        if suitable_values != (0.0,):
            raise ValueError("面积口径当前仅支持 suitable_values=0（与 v7/v10 默认一致）")

        wind = _read_csv_auto_encoding(str(args.wind_csv))
        solar = _read_csv_auto_encoding(str(args.solar_csv))

        # keep valid provinces (与 diagnose_site_distribution_v5.py 口径一致)
        wind = wind[~wind["province"].isin(["unknown", "境界线"])].copy()
        solar = solar[~solar["province"].isin(["unknown", "境界线"])].copy()

        for df in (wind, solar):
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df.dropna(subset=["lon", "lat"], inplace=True)
            df["lon"] = df["lon"].astype(float)
            df["lat"] = df["lat"].astype(float)
            df["province"] = df["province"].astype(str)

        def _anchor_years(df: pd.DataFrame) -> List[int]:
            ys = []
            for c in df.columns:
                if not c.startswith("capacity_"):
                    continue
                try:
                    ys.append(int(c.split("_")[-1]))
                except Exception:
                    pass
            ys = sorted(set(ys))
            ys = [y for y in ys if int(args.min_year) <= y <= int(args.max_year)]
            return ys

        years_w = _anchor_years(wind)
        years_s = _anchor_years(solar)
        years_anchor = sorted(set(years_w) & set(years_s)) or sorted(set(years_w) | set(years_s))

        # 只对“可能用到”的网格计算面积占比（避免对 15 万行做逐窗口 IO）
        cap_cols = [f"capacity_{int(y)}" for y in years_anchor if f"capacity_{int(y)}" in wind.columns]
        w_need = np.zeros((len(wind),), dtype=bool)
        if cap_cols:
            w_need = (wind[cap_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0).any(axis=1)
        s_need = np.zeros((len(solar),), dtype=bool)
        cap_cols_s = [f"capacity_{int(y)}" for y in years_anchor if f"capacity_{int(y)}" in solar.columns]
        if cap_cols_s:
            s_need = (solar[cap_cols_s].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0).any(axis=1)

        frac_w = np.full((len(wind),), np.nan, dtype=np.float32)
        if int(np.count_nonzero(w_need)) > 0:
            if str(args.area_method) == "lonlat0p1":
                frac_w[w_need] = compute_suitable_area_fraction_by_lonlat_cell(
                    str(args.wind_mask),
                    wind.loc[w_need, "lon"].to_numpy(dtype=float),
                    wind.loc[w_need, "lat"].to_numpy(dtype=float),
                    half_deg=0.05,
                    suitable_values=(0.0,),
                    nodata_value=-1.0,
                )
            else:
                frac_w[w_need] = compute_suitable_area_fraction_by_window(
                    str(args.wind_mask),
                    wind.loc[w_need, "lon"].to_numpy(dtype=float),
                    wind.loc[w_need, "lat"].to_numpy(dtype=float),
                    half_size_m=5000.0,
                    suitable_values=(0.0,),
                    nodata_value=-1.0,
                )

        frac_s = np.full((len(solar),), np.nan, dtype=np.float32)
        if int(np.count_nonzero(s_need)) > 0:
            if str(args.area_method) == "lonlat0p1":
                frac_s[s_need] = compute_suitable_area_fraction_by_lonlat_cell(
                    str(args.solar_mask),
                    solar.loc[s_need, "lon"].to_numpy(dtype=float),
                    solar.loc[s_need, "lat"].to_numpy(dtype=float),
                    half_deg=0.05,
                    suitable_values=(0.0,),
                    nodata_value=-1.0,
                )
            else:
                frac_s[s_need] = compute_suitable_area_fraction_by_window(
                    str(args.solar_mask),
                    solar.loc[s_need, "lon"].to_numpy(dtype=float),
                    solar.loc[s_need, "lat"].to_numpy(dtype=float),
                    half_size_m=5000.0,
                    suitable_values=(0.0,),
                    nodata_value=-1.0,
                )

        wind_yearly_anchor, wind_prov_2024 = _compute_share(wind, years_anchor, frac_w)
        solar_yearly_anchor, solar_prov_2024 = _compute_share(solar, years_anchor, frac_s)

        wind_yearly_anchor.to_csv(os.path.join(cache_dir, f"stats_wind_suitable_area_share_anchor_years_{tag}.csv"), index=False)
        solar_yearly_anchor.to_csv(os.path.join(cache_dir, f"stats_solar_suitable_area_share_anchor_years_{tag}.csv"), index=False)
        # 兼容旧文件名（2024）；若 2024 也在 province-years 中，随后会被带 total_capacity 的表覆盖
        wind_prov_2024.to_csv(os.path.join(cache_dir, f"stats_wind_suitable_area_share_2024_by_province_{tag}.csv"), index=False)
        solar_prov_2024.to_csv(os.path.join(cache_dir, f"stats_solar_suitable_area_share_2024_by_province_{tag}.csv"), index=False)
        for yy in prov_years:
            wind_prov_Y = _compute_province_share_for_year(wind, yy, frac_w)
            solar_prov_Y = _compute_province_share_for_year(solar, yy, frac_s)
            wind_prov_Y.to_csv(os.path.join(cache_dir, f"stats_wind_suitable_area_share_{yy}_by_province_{tag}.csv"), index=False)
            solar_prov_Y.to_csv(os.path.join(cache_dir, f"stats_solar_suitable_area_share_{yy}_by_province_{tag}.csv"), index=False)
        print(f"[OK] wrote stats to {cache_dir}")

    if do_plot:
        saved = _plot_all_from_tables(
            cache_dir=cache_dir,
            out_dir=out_dir,
            tag=tag,
            prov_years=prov_years,
            abbr_map=abbr_map,
        )
        print("Saved:")
        for p in saved:
            print(" -", p)


if __name__ == "__main__":
    main()
