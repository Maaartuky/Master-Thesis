"""LOD/LOQ analysis for BPE spectra using integrated peak area.

This module can be used in two ways:
1) As a CLI script.
2) Copy/paste in Jupyter Notebook and call `run_lod_loq_analysis(...)`.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PAT_BPE = re.compile(r"^BPE_(\d+(?:\.\d+)?)([mun]?M)_auto_sam(\d+)_mes(\d+)\.csv$", re.IGNORECASE)
PAT_BLANK = re.compile(r"^EtOH_auto_sam(\d+)\.csv$", re.IGNORECASE)


@dataclass(frozen=True)
class PeakConfig:
    center_cm1: float = 1635
    halfwidth_cm1: float = 8
    left_near_cm1: float = 8
    left_far_cm1: float = 18
    right_near_cm1: float = 8
    right_far_cm1: float = 18


@dataclass(frozen=True)
class LODLOQResult:
    slope: float
    intercept: float
    r2: float
    sigma_blank: float
    lod_m: float
    loq_m: float


def conc_to_molar(value: str, unit: str) -> float:
    factor_map = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}
    key = unit.lower()
    if key not in factor_map:
        raise ValueError(f"Unidad no reconocida: {unit}")
    return float(value) * factor_map[key]


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None, names=["shift_cm1", "intensity_au"]).dropna()
    df["shift_cm1"] = pd.to_numeric(df["shift_cm1"], errors="coerce")
    df["intensity_au"] = pd.to_numeric(df["intensity_au"], errors="coerce")
    df = df.dropna().sort_values("shift_cm1")
    return df["shift_cm1"].to_numpy(), df["intensity_au"].to_numpy()


def integrated_peak_area_anchor(shift_cm1: np.ndarray, intensity_au: np.ndarray, cfg: PeakConfig) -> float:
    sh = np.asarray(shift_cm1)
    it = np.asarray(intensity_au)

    m_left = (sh >= cfg.center_cm1 - cfg.left_far_cm1) & (sh <= cfg.center_cm1 - cfg.left_near_cm1)
    m_right = (sh >= cfg.center_cm1 + cfg.right_near_cm1) & (sh <= cfg.center_cm1 + cfg.right_far_cm1)
    if m_left.sum() < 3 or m_right.sum() < 3:
        return np.nan

    x_left, y_left = sh[m_left].mean(), it[m_left].mean()
    x_right, y_right = sh[m_right].mean(), it[m_right].mean()
    if np.isclose(x_left, x_right):
        return np.nan

    slope = (y_right - y_left) / (x_right - x_left)
    intercept = y_left - slope * x_left

    m_peak = (sh >= cfg.center_cm1 - cfg.halfwidth_cm1) & (sh <= cfg.center_cm1 + cfg.halfwidth_cm1)
    if m_peak.sum() < 3:
        return np.nan

    baseline = slope * sh[m_peak] + intercept
    corrected = it[m_peak] - baseline
    return float(np.trapz(corrected, sh[m_peak]))


def parse_records(base_dir: Path) -> tuple[list[dict], list[dict]]:
    bpe_records: list[dict] = []
    blank_records: list[dict] = []

    for csv_path in sorted(base_dir.glob("*.csv")):
        bpe_match = PAT_BPE.match(csv_path.name)
        if bpe_match:
            value, unit, sam, mes = bpe_match.groups()
            shift, intensity = load_csv(csv_path)
            bpe_records.append(
                {
                    "file": csv_path.name,
                    "conc_m": conc_to_molar(value, unit),
                    "conc_label": f"{value}{unit}",
                    "sam": int(sam),
                    "mes": int(mes),
                    "shift": shift,
                    "int": intensity,
                }
            )
            continue

        blank_match = PAT_BLANK.match(csv_path.name)
        if blank_match:
            sam = blank_match.group(1)
            shift, intensity = load_csv(csv_path)
            blank_records.append({"file": csv_path.name, "sam": int(sam), "shift": shift, "int": intensity})

    if not bpe_records:
        raise RuntimeError(f"No se encontraron archivos BPE válidos en {base_dir}")
    if not blank_records:
        raise RuntimeError(f"No se encontraron archivos de blanco EtOH en {base_dir}")

    return bpe_records, blank_records


def summarize_areas(records: Iterable[dict], cfg: PeakConfig, area_col: str) -> pd.DataFrame:
    rows = []
    for record in records:
        area = integrated_peak_area_anchor(record["shift"], record["int"], cfg)
        row = {k: v for k, v in record.items() if k not in {"shift", "int"}}
        row[area_col] = area
        rows.append(row)
    return pd.DataFrame(rows)


def compute_lod_loq(
    df_blank: pd.DataFrame,
    df_bpe: pd.DataFrame,
    exclude_conc_m: float,
    k_lod: float = 3.3,
    k_loq: float = 10.0,
) -> tuple[LODLOQResult, pd.DataFrame]:
    sigma_blank = float(df_blank["area_blank"].std(ddof=1)) if len(df_blank) > 1 else 0.0

    grouped = (
        df_bpe.dropna(subset=["area"])
        .groupby(["conc_m", "conc_label"], as_index=False)
        .agg(mean_area=("area", "mean"), std_area=("area", "std"), n=("area", "size"))
    )

    fit_df = grouped[grouped["conc_m"] < exclude_conc_m].dropna(subset=["mean_area"])
    if len(fit_df) < 2:
        raise RuntimeError("No hay suficientes puntos para el ajuste lineal tras aplicar el filtro de concentración")

    x = fit_df["conc_m"].to_numpy(dtype=float)
    y = fit_df["mean_area"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    if np.isclose(slope, 0.0):
        raise RuntimeError("Pendiente ~0: no se puede calcular LOD/LOQ de forma estable")

    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    lod_m = k_lod * sigma_blank / slope
    loq_m = k_loq * sigma_blank / slope

    return (
        LODLOQResult(
            slope=float(slope),
            intercept=float(intercept),
            r2=float(r2),
            sigma_blank=sigma_blank,
            lod_m=float(lod_m),
            loq_m=float(loq_m),
        ),
        grouped,
    )


def run_lod_loq_analysis(
    base_dir: str | Path = ".",
    peak_center_cm1: float = 1635,
    integrate_halfwidth_cm1: float = 8,
    left_near_cm1: float = 8,
    left_far_cm1: float = 18,
    right_near_cm1: float = 8,
    right_far_cm1: float = 18,
    exclude_conc_m: float = 1e-3,
    k_lod: float = 3.3,
    k_loq: float = 10.0,
) -> dict:
    """Notebook-friendly entrypoint.

    Returns a dict with dataframes and final LOD/LOQ results.
    """
    cfg = PeakConfig(
        center_cm1=peak_center_cm1,
        halfwidth_cm1=integrate_halfwidth_cm1,
        left_near_cm1=left_near_cm1,
        left_far_cm1=left_far_cm1,
        right_near_cm1=right_near_cm1,
        right_far_cm1=right_far_cm1,
    )
    base_path = Path(base_dir)
    bpe_records, blank_records = parse_records(base_path)

    df_blank = summarize_areas(blank_records, cfg, area_col="area_blank").sort_values("sam")
    df_bpe = summarize_areas(bpe_records, cfg, area_col="area").sort_values(["conc_m", "sam", "mes"])
    result, grouped = compute_lod_loq(
        df_blank=df_blank,
        df_bpe=df_bpe,
        exclude_conc_m=exclude_conc_m,
        k_lod=k_lod,
        k_loq=k_loq,
    )

    return {
        "config": cfg,
        "df_blank": df_blank,
        "df_bpe": df_bpe,
        "df_summary": grouped,
        "result": result,
    }


def print_report(output: dict) -> None:
    result: LODLOQResult = output["result"]
    df_blank = output["df_blank"]
    df_bpe = output["df_bpe"]

    print("=== Resumen blanco ===")
    print(df_blank[["file", "sam", "area_blank"]].to_string(index=False))
    print("\n=== Resumen BPE (primeras 10 filas) ===")
    print(df_bpe[["file", "conc_label", "conc_m", "sam", "mes", "area"]].head(10).to_string(index=False))

    print("\n=== Ajuste lineal y LOD/LOQ ===")
    print(f"sigma_blank = {result.sigma_blank:.6g} a.u.*cm^-1")
    print(f"m = {result.slope:.6g} (a.u.*cm^-1)/M")
    print(f"b = {result.intercept:.6g} a.u.*cm^-1")
    print(f"R2 = {result.r2:.6g}")
    print(f"LOD = {result.lod_m:.6g} M ({result.lod_m / 1e-6:.3g} uM)")
    print(f"LOQ = {result.loq_m:.6g} M ({result.loq_m / 1e-6:.3g} uM)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calcular LOD/LOQ de BPE en EtOH por área integrada")
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Carpeta con los CSV")
    parser.add_argument("--peak-center", type=float, default=1635)
    parser.add_argument("--halfwidth", type=float, default=8)
    parser.add_argument("--exclude-conc-m", type=float, default=1e-3, help="Excluir concentraciones >= este valor para ajuste")
    # parse_known_args evita que Jupyter (ipykernel) rompa por argumentos extra como "-f ...json"
    args, _unknown = parser.parse_known_args(argv)

    output = run_lod_loq_analysis(
        base_dir=args.base_dir,
        peak_center_cm1=args.peak_center,
        integrate_halfwidth_cm1=args.halfwidth,
        exclude_conc_m=args.exclude_conc_m,
    )
    print_report(output)


if __name__ == "__main__":
    main()
