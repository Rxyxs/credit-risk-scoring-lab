"""Graficos de resultados: EDA de WOE/IV, diagnostico del scorecard (KS,
distribucion de score, PSI), comparacion scorecard-R vs ML-Python, SHAP,
y benchmark del motor en C. Misma paleta validada (accesible, orden
categorico fijo) usada en el resto del portafolio.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

CAT = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948",
}

MODEL_COLORS = {
    "scorecard_r": CAT["blue"],
    "logistic_regression": CAT["magenta"],
    "random_forest": CAT["aqua"],
    "xgboost": CAT["yellow"],
    "lightgbm": CAT["violet"],
    "deep_learning": CAT["green"],
}
MODEL_LABELS = {
    "scorecard_r": "Scorecard R (WOE + Logit)",
    "logistic_regression": "Regresion Logistica",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "deep_learning": "MLP (PyTorch, Focal Loss)",
}
ACTIVATION_COLORS = {"relu": CAT["red"], "gelu": CAT["orange"], "swish": CAT["green"]}


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_iv_summary(iv_csv: Path, out_path: Path):
    df = pd.read_csv(iv_csv).sort_values("iv")
    colors = [CAT["blue"] if iv >= 0.02 else INK_MUTED for iv in df["iv"]]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    bars = ax.barh(df["feature"], df["iv"], color=colors, zorder=3)
    for bar, val in zip(bars, df["iv"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                 va="center", fontsize=8, color=INK_SECONDARY)
    ax.axvline(0.02, color=INK_MUTED, linewidth=1, linestyle="--", zorder=2)
    ax.text(0.02, -0.7, "umbral IV=0.02", fontsize=7, color=INK_MUTED, ha="left")

    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Information Value (IV)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Poder predictivo por variable (WOE/IV, train)", color=INK_PRIMARY,
                 fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_woe_bins(woe_csv: Path, out_path: Path, features: list[str]):
    df = pd.read_csv(woe_csv)
    fig, axes = plt.subplots(1, len(features), figsize=(4.2 * len(features), 4.5), facecolor=SURFACE)
    for ax, feat in zip(axes, features):
        sub = df[df["feature"] == feat].sort_values("bin_id")
        colors = [CAT["red"] if w < 0 else CAT["blue"] for w in sub["woe"]]
        ax.bar(sub["bin_id"].astype(str), sub["woe"], color=colors, zorder=3)
        ax.axhline(0, color=INK_MUTED, linewidth=1, zorder=2)
        _style_axes(ax)
        ax.set_xlabel("bin (0 = valores mas bajos)", color=INK_SECONDARY, fontsize=8)
        ax.set_title(feat, color=INK_PRIMARY, fontsize=10, fontweight="bold")
    axes[0].set_ylabel("WOE", color=INK_SECONDARY, fontsize=9)
    fig.suptitle("Monotonicidad del WOE por bin (mayor WOE = mas seguro)", color=INK_PRIMARY,
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_ks_chart(score_dist_csv: Path, out_path: Path):
    df = pd.read_csv(score_dist_csv)
    test = df[df["split"] == "test"].sort_values("score")
    n_good = (test["default_12m"] == 0).sum()
    n_bad = (test["default_12m"] == 1).sum()
    cum_good = (test["default_12m"] == 0).cumsum() / n_good
    cum_bad = (test["default_12m"] == 1).cumsum() / n_bad
    diff = np.abs(cum_bad.to_numpy() - cum_good.to_numpy())
    ks_idx = int(np.argmax(diff))
    ks_stat = diff[ks_idx]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    ax.plot(test["score"].to_numpy(), cum_good.to_numpy(), color=CAT["blue"], linewidth=2, label="% acumulado buenos")
    ax.plot(test["score"].to_numpy(), cum_bad.to_numpy(), color=CAT["red"], linewidth=2, label="% acumulado malos")
    ax.plot(
        [test["score"].to_numpy()[ks_idx]] * 2, [cum_good.to_numpy()[ks_idx], cum_bad.to_numpy()[ks_idx]],
        color=INK_PRIMARY, linewidth=1.5, linestyle="--",
    )
    ax.text(test["score"].to_numpy()[ks_idx], (cum_good.to_numpy()[ks_idx] + cum_bad.to_numpy()[ks_idx]) / 2,
            f"  KS = {ks_stat:.3f}", fontsize=10, color=INK_PRIMARY, fontweight="bold", va="center")

    _style_axes(ax)
    ax.set_xlabel("Score", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("% acumulado", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Curva KS (Kolmogorov-Smirnov) — scorecard sobre test", color=INK_PRIMARY,
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.92, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_score_distribution(score_dist_csv: Path, out_path: Path):
    df = pd.read_csv(score_dist_csv)
    test = df[df["split"] == "test"]
    good = test[test["default_12m"] == 0]["score"]
    bad = test[test["default_12m"] == 1]["score"]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    bins = np.linspace(df["score"].min(), df["score"].max(), 40)
    ax.hist(good, bins=bins, color=CAT["blue"], alpha=0.65, density=True, label=f"Buenos (n={len(good)})", zorder=3)
    ax.hist(bad, bins=bins, color=CAT["red"], alpha=0.65, density=True, label=f"Malos (n={len(bad)})", zorder=3)

    _style_axes(ax)
    ax.set_xlabel("Score", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Densidad", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Distribucion del score por resultado real (test)", color=INK_PRIMARY,
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.92, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_model_comparison(r_metrics_json: Path, ml_metrics_json: Path, out_path: Path, dl_metrics_json: Path | None = None):
    r_metrics = json.loads(r_metrics_json.read_text())
    ml_metrics = json.loads(ml_metrics_json.read_text())["metrics"]

    rows = [{"model": "scorecard_r", "auc": r_metrics["auc"], "gini": r_metrics["gini"], "ks": r_metrics["ks_statistic"]}]
    for name, m in ml_metrics.items():
        rows.append({"model": name, "auc": m["test"]["auc"], "gini": m["test"]["gini"], "ks": m["test"]["ks_statistic"]})
    if dl_metrics_json is not None and dl_metrics_json.exists():
        dl_report = json.loads(dl_metrics_json.read_text())
        best = dl_report["metrics"][dl_report["best_activation"]]
        rows.append({"model": "deep_learning", "auc": best["auc"], "gini": best["gini"], "ks": best["ks_statistic"]})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=SURFACE)
    for ax, metric, title in zip(axes, ["auc", "gini", "ks"], ["AUC", "Gini", "KS"]):
        colors = [MODEL_COLORS[m] for m in df["model"]]
        bars = ax.bar(range(len(df)), df[metric], color=colors, zorder=3)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}",
                     ha="center", fontsize=8, color=INK_SECONDARY)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in df["model"]], rotation=35, ha="right", fontsize=7.5)
        _style_axes(ax)
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")

    fig.suptitle("Scorecard estadistico (R) vs. modelos ML (Python) — holdout de test",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_psi(psi_test_csv: Path, psi_drifted_csv: Path, metrics_json: Path, out_path: Path):
    test_bins = pd.read_csv(psi_test_csv)
    drifted_bins = pd.read_csv(psi_drifted_csv)
    metrics = json.loads(metrics_json.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE)
    for ax, bins_df, title, psi_val in zip(
        axes, [test_bins, drifted_bins],
        [f"Test vs. train\nPSI = {metrics['psi_test_vs_train']:.4f}",
         f"Poblacion 'drifted' vs. train\nPSI = {metrics['psi_drifted_vs_train']:.4f}"],
        [metrics["psi_test_vs_train"], metrics["psi_drifted_vs_train"]],
    ):
        x = np.arange(len(bins_df))
        width = 0.38
        ax.bar(x - width / 2, bins_df["pct_baseline"], width, color=CAT["blue"], label="Esperado (train)", zorder=3)
        ax.bar(x + width / 2, bins_df["pct_actual"], width, color=CAT["orange"], label="Actual", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)
        _style_axes(ax)
        ax.set_xlabel("Decil de score", color=INK_SECONDARY, fontsize=9)
        ax.set_title(title, color=INK_PRIMARY, fontsize=10.5, fontweight="bold")
        ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.92, fontsize=8, labelcolor=INK_SECONDARY)
    axes[0].set_ylabel("% de la poblacion", color=INK_SECONDARY, fontsize=9)

    fig.suptitle("PSI (Population Stability Index) — monitoreo de drift del score",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_shap_importance(shap_csv: Path, out_path: Path, top_n: int = 12):
    df = pd.read_csv(shap_csv).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    bars = ax.barh(df["feature"], df["mean_abs_shap"], color=CAT["violet"], zorder=3)
    for bar, val in zip(bars, df["mean_abs_shap"]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                 va="center", fontsize=8, color=INK_SECONDARY)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value|", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Explicabilidad SHAP — mejor modelo ML", color=INK_PRIMARY,
                 fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_dl_loss_curves(loss_curve_csv: Path, out_path: Path):
    df = pd.read_csv(loss_curve_csv)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE)

    for activation, sub in df.groupby("activation"):
        color = ACTIVATION_COLORS.get(activation, INK_MUTED)
        axes[0].plot(sub["epoch"], sub["train_loss"], color=color, linewidth=1.6, linestyle="--", alpha=0.7)
        axes[0].plot(sub["epoch"], sub["val_loss"], color=color, linewidth=2, label=activation)
        axes[1].plot(sub["epoch"], sub["val_auc"], color=color, linewidth=2, label=activation)

    for ax, title, ylabel in zip(
        axes, ["Focal Loss por epoca (— val, -- train)", "AUC en test por epoca"], ["Focal loss", "AUC"]
    ):
        _style_axes(ax)
        ax.set_xlabel("Epoca", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")
        ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.92, fontsize=8, labelcolor=INK_SECONDARY)

    fig.suptitle("MLP (PyTorch) — comparacion de activaciones (ReLU vs GELU vs Swish/SiLU)",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_dl_loss_curves_animated(loss_curve_csv: Path, out_path: Path, max_frames: int = 60):
    """GIF 'racing line chart' de las mismas curvas de perdida/AUC por epoca
    (misma fuente de datos que plot_dl_loss_curves), con etiqueta flotante
    en la punta de cada linea mostrando el valor actual.
    """
    df = pd.read_csv(loss_curve_csv)
    activations = sorted(df["activation"].unique())
    epochs = sorted(df["epoch"].unique())
    n_epochs = len(epochs)
    if n_epochs > max_frames:
        idx = np.linspace(0, n_epochs - 1, max_frames).astype(int)
        frame_epochs = [epochs[i] for i in sorted(set(idx))]
    else:
        frame_epochs = epochs

    series = {
        act: df[df["activation"] == act].sort_values("epoch").reset_index(drop=True)
        for act in activations
    }

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    x_max = max(epochs)
    loss_min = df["val_loss"].min() * 0.9
    loss_max = df["train_loss"].max() * 1.1
    auc_min = df["val_auc"].min() * 0.98
    auc_max = df["val_auc"].max() * 1.02

    axes[0].set_xlim(1, x_max)
    axes[0].set_ylim(loss_min, loss_max)
    axes[0].set_title("Focal Loss por epoca (val)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Focal loss")

    axes[1].set_xlim(1, x_max)
    axes[1].set_ylim(auc_min, auc_max)
    axes[1].set_title("AUC en test por epoca", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("AUC")

    fig.suptitle(
        "MLP (PyTorch) — evolucion del entrenamiento por activacion (ReLU vs GELU vs Swish/SiLU)",
        fontsize=12, fontweight="bold",
    )

    lines_loss, lines_auc, labels_loss, labels_auc = {}, {}, {}, {}
    for act in activations:
        color = ACTIVATION_COLORS.get(act, "#cccccc")
        (line_l,) = axes[0].plot([], [], color=color, linewidth=2, label=act)
        (line_a,) = axes[1].plot([], [], color=color, linewidth=2, label=act)
        lines_loss[act], lines_auc[act] = line_l, line_a
        labels_loss[act] = axes[0].annotate(
            "", xy=(1, 0), xytext=(8, 8), textcoords="offset points",
            fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.9),
        )
        labels_auc[act] = axes[1].annotate(
            "", xy=(1, 0), xytext=(8, 8), textcoords="offset points",
            fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.9),
        )

    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    def update(frame_idx):
        current_epoch = frame_epochs[frame_idx]
        artists = []
        for act in activations:
            sub = series[act]
            mask = sub["epoch"] <= current_epoch
            x = sub.loc[mask, "epoch"].to_numpy()
            y_loss = sub.loc[mask, "val_loss"].to_numpy()
            y_auc = sub.loc[mask, "val_auc"].to_numpy()

            lines_loss[act].set_data(x, y_loss)
            lines_auc[act].set_data(x, y_auc)

            if len(x):
                labels_loss[act].xy = (x[-1], y_loss[-1])
                labels_loss[act].set_text(f"{act}: {y_loss[-1]:.4f}")
                labels_auc[act].xy = (x[-1], y_auc[-1])
                labels_auc[act].set_text(f"{act}: {y_auc[-1]:.4f}")

            artists.extend([lines_loss[act], lines_auc[act], labels_loss[act], labels_auc[act]])
        return artists

    ani = animation.FuncAnimation(fig, update, frames=len(frame_epochs), interval=120, blit=False)
    ani.save(out_path, writer="pillow")
    plt.close(fig)
    plt.style.use("default")


def plot_c_benchmark(benchmark_json: Path, out_path: Path):
    results = json.loads(benchmark_json.read_text())
    tp = results["throughput"]
    labels = ["Python puro", "NumPy vectorizado", "C (ctypes)"]
    values = [
        tp["python_pure_loop"]["rows_per_sec"],
        tp["numpy_vectorized"]["rows_per_sec"],
        tp["c_engine"]["rows_per_sec"],
    ]
    colors = [CAT["red"], CAT["orange"], CAT["blue"]]

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    bars = ax.bar(labels, values, color=colors, zorder=3)
    ax.set_yscale("log")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15, f"{val:,.0f}",
                 ha="center", fontsize=9, color=INK_SECONDARY)
    _style_axes(ax)
    ax.grid(axis="y", which="both", color=GRID, linewidth=0.6, zorder=0)
    ax.set_ylabel("Filas/segundo (escala log)", color=INK_SECONDARY, fontsize=10)
    ax.set_title(f"Motor de scoring: throughput sobre {results['n_rows_benchmarked']:,} filas",
                 color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def generate_all_plots(base_dir: Path):
    reports = base_dir / "outputs" / "reports"
    plots_dir = base_dir / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_iv_summary(reports / "iv_summary.csv", plots_dir / "iv_summary.png")
    plot_woe_bins(
        reports / "woe_bins.csv", plots_dir / "woe_bins.png",
        features=["tipo_contrato", "n_morosidad_reportes", "renta_liquida", "dti"],
    )
    plot_ks_chart(reports / "score_distribution.csv", plots_dir / "ks_chart.png")
    plot_score_distribution(reports / "score_distribution.csv", plots_dir / "score_distribution.png")
    plot_model_comparison(
        reports / "scorecard_validation_metrics.json", reports / "ml_model_comparison.json",
        plots_dir / "model_comparison.png", dl_metrics_json=reports / "dl_model_comparison.json",
    )
    plot_psi(
        reports / "psi_bins_test.csv", reports / "psi_bins_drifted.csv",
        reports / "scorecard_validation_metrics.json", plots_dir / "psi_monitoring.png",
    )
    plot_shap_importance(reports / "shap_importance.csv", plots_dir / "shap_importance.png")
    plot_c_benchmark(reports / "c_engine_benchmark.json", plots_dir / "c_engine_benchmark.png")
    if (reports / "dl_loss_curves.csv").exists():
        plot_dl_loss_curves(reports / "dl_loss_curves.csv", plots_dir / "dl_loss_curves.png")
        plot_dl_loss_curves_animated(reports / "dl_loss_curves.csv", plots_dir / "dl_loss_curves_animated.gif")
    print(f"Graficos guardados en {plots_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    generate_all_plots(base_dir)
