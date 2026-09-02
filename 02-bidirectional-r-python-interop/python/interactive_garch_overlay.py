"""
Interactive Plotly chart: candlestick OHLC price series with the fitted
GARCH(1,1) conditional volatility overlaid, colored by volatility regime
(Baja/Media/Alta terciles) -- the same regime classification used by
r/volatility_garch.R and consumed downstream by the combined credit/market
risk heatmap (r/run_combined_analysis.R). Self-contained HTML, inline JS.

Usage:
    .venv\\Scripts\\python.exe -m python.interactive_garch_overlay
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

TABLE_PATH = Path("output/tables/market_volatility_regimes.csv")
OUT_DIR = Path("output/interactive")

REGIME_COLORS = {"Baja": "#2E6E62", "Media": "#C9A227", "Alta": "#8B2E2E"}


def main() -> None:
    df = pd.read_csv(TABLE_PATH, parse_dates=["fecha"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.06,
        subplot_titles=(
            "Synthetic OHLC price (GARCH(1,1)-driven returns)",
            "GARCH(1,1) conditional volatility, by regime (terciles)",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=df["fecha"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#2E6E62",
            decreasing_line_color="#8B2E2E",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    for regime in ["Baja", "Media", "Alta"]:
        mask = df["regimen_volatilidad"] == regime
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, "fecha"],
                y=df.loc[mask, "volatilidad_estimada_garch"],
                mode="markers",
                name=f"Regime: {regime}",
                marker=dict(color=REGIME_COLORS[regime], size=5),
                hovertemplate=(
                    f"Regime {regime}<br>%{{x|%Y-%m-%d}}<br>"
                    "GARCH vol: %{y:.4f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["volatilidad_estimada_garch"],
            mode="lines",
            line=dict(color="#1F4E79", width=1),
            opacity=0.4,
            name="GARCH vol (line)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    n_by_regime = df["regimen_volatilidad"].value_counts()
    subtitle = " / ".join(f"{r}: {n_by_regime.get(r, 0)}d" for r in ["Baja", "Media", "Alta"])

    fig.update_layout(
        title=(
            "Market volatility regimes overlaid on price -- GARCH(1,1), 750 "
            f"trading days<br><sup>Regime day counts -- {subtitle}. Feeds the "
            "combined credit-risk x market-risk stressed-loss heatmap (see README).</sup>"
        ),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
        font=dict(family="Segoe UI, Arial, sans-serif", size=13),
        xaxis_rangeslider_visible=False,
        margin=dict(t=120),
        height=760,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Conditional volatility", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "garch_volatility_regime_overlay.html"
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    print(f"Interactive chart saved -> {out_path}")
    print(n_by_regime)


if __name__ == "__main__":
    main()
