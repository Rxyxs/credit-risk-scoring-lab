"""
Genera un grafico Plotly interactivo (HTML standalone, inline JS) con la
distribucion del score del scorecard R por banda de riesgo (deciles),
superpuesta con la tasa de default observada en cada banda -- el mismo
mecanismo de concentracion de riesgo citado en el README (lift 3.33x en
el peor decil, 52% de los defaults capturados rechazando el 20% peor).

Uso:
    python -m src.visualization.interactive_score_distribution
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

REPORTS_DIR = Path("outputs/reports")
OUT_DIR = Path("outputs/interactive")


def main() -> None:
    df = pd.read_csv(REPORTS_DIR / "score_distribution.csv")
    test = df[df["split"] == "test"].copy()

    # Bandas de riesgo: deciles de score, banda 1 = score mas bajo (mas riesgoso)
    test["risk_decile"] = pd.qcut(test["score"], 10, labels=False, duplicates="drop") + 1
    band_stats = (
        test.groupby("risk_decile")
        .agg(
            n=("default_12m", "size"),
            bad_rate=("default_12m", "mean"),
            score_min=("score", "min"),
            score_max=("score", "max"),
        )
        .reset_index()
    )
    band_stats["bad_rate_pct"] = band_stats["bad_rate"] * 100
    portfolio_bad_rate = test["default_12m"].mean() * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colors = ["#8B2E2E" if d <= 2 else "#C9A227" if d <= 5 else "#2E6E62" for d in band_stats["risk_decile"]]

    fig.add_trace(
        go.Histogram(
            x=test.loc[test["default_12m"] == 0, "score"],
            name="No default (12m)",
            marker_color="#2E6E62",
            opacity=0.55,
            nbinsx=40,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Histogram(
            x=test.loc[test["default_12m"] == 1, "score"],
            name="Default (12m)",
            marker_color="#8B2E2E",
            opacity=0.75,
            nbinsx=40,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=[(row.score_min + row.score_max) / 2 for row in band_stats.itertuples()],
            y=band_stats["bad_rate_pct"],
            mode="lines+markers",
            name="Bad rate by decile (%)",
            line=dict(color="#1F4E79", width=3),
            marker=dict(size=9, color=colors, line=dict(color="#1F4E79", width=1)),
            hovertext=[
                f"Decile {row.risk_decile} (worst=1)<br>Score {row.score_min:.0f}-{row.score_max:.0f}"
                f"<br>n={row.n}<br>Bad rate {row.bad_rate_pct:.2f}%"
                for row in band_stats.itertuples()
            ],
            hoverinfo="text",
        ),
        secondary_y=True,
    )

    fig.add_hline(
        y=portfolio_bad_rate,
        line_dash="dash",
        line_color="#666666",
        annotation_text=f"Portfolio avg bad rate: {portfolio_bad_rate:.2f}%",
        annotation_position="top left",
        secondary_y=True,
    )

    fig.update_layout(
        title=(
            "Scorecard score distribution by risk decile -- test set (n="
            f"{len(test):,})<br><sup>R WOE/logit scorecard, seed 42 -- worst "
            "decile bad-rate lift vs. portfolio average, see README section 3</sup>"
        ),
        barmode="overlay",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        font=dict(family="Segoe UI, Arial, sans-serif", size=13),
        margin=dict(t=110),
    )
    fig.update_xaxes(title_text="Scorecard score (points)")
    fig.update_yaxes(title_text="Applicant count", secondary_y=False)
    fig.update_yaxes(title_text="12-month default rate (%)", secondary_y=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "score_distribution_by_risk_band.html"
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    print(f"Interactive chart saved -> {out_path}")
    print(band_stats.to_string(index=False))


if __name__ == "__main__":
    main()
