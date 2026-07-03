from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ReportOutputDirs:
    output_dir: Path
    figure_dir: Path
    table_dir: Path


def create_report_output_dirs(
        project_root: Path,
        notebook_name: str,
) -> ReportOutputDirs:
    output_dir = project_root / "Outputs" / notebook_name
    figure_dir = output_dir / "Figures"
    table_dir = output_dir / "Tables"

    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    return ReportOutputDirs(
        output_dir=output_dir,
        figure_dir=figure_dir,
        table_dir=table_dir,
    )


def set_report_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


def save_report_figure(
        fig,
        figure_dir: Path,
        file_stem: str,
        dpi: int = 300,
) -> Path:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    png_path = figure_dir / f"{file_stem}.png"

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Saved figure: {png_path}")

    return png_path


def export_report_table(
        table: pd.DataFrame,
        table_dir: Path,
        file_stem: str,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        index: bool = False,
        float_format: str = "%.2f",
        escape: bool = False,
) -> dict[str, Path]:
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    tex_path = table_dir / f"{file_stem}.tex"

    table.to_latex(
        tex_path,
        index=index,
        caption=caption,
        label=label,
        escape=escape,
        float_format=float_format,
    )

    print(f"Saved table: {tex_path}")

    return {"tex": tex_path}
