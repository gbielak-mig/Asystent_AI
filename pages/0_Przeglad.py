"""
GA4 Przegląd
============
Strona startowa: BEZ udziału LLM/Groq — deterministyczne, liczone bezpośrednio
w kodzie sekcje:
  - Trendy: który rynek rośnie/spada (wybrana metryka), wykres do rozwinięcia,
  - Nowe analizy: automatycznie wykryte anomalie (przychód, ruch, konwersje,
    CR, wsp. odbić, porzucone koszyki) względem 30-dniowej historii.

Nic tu nie zależy od limitów Groq — więc działa zawsze, niezależnie od stanu
czatu na stronie głównej.
"""

from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

import ga4_core as core

st.set_page_config(page_title="GA4 Przegląd", page_icon="📊", layout="wide")
core.require_auth()

st.title("📊 Przegląd")
st.caption(
    "Trendy per rynek i automatycznie wykryte anomalie — liczone bezpośrednio "
    "w kodzie, bez udziału czatu/Groq."
)

# ─────────────────────────────────────────────────────────────
# USTAWIENIA
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Ustawienia")
    lookback_days = st.slider("Okres do wykresów trendu (dni)", 14, 90, 30)
    trend_metric = st.selectbox(
        "Metryka do sekcji Trendy", core.OVERVIEW_METRICS,
        format_func=lambda m: core.METRIC_LABELS.get(m, m),
    )
    move_threshold = st.slider(
        "Próg 'rosnący/spadający' (% zmiany tydz./tydz.)", 1, 30, 5,
    )
    sigma_threshold = st.slider(
        "Próg anomalii (σ) w sekcji Nowe analizy", 1.0, 4.0, 2.0, step=0.5,
    )
    brand_options = sorted(core.property_map["Brand"].unique())
    brand_filter = st.multiselect(
        "Ogranicz do brandów (opcjonalnie)", brand_options,
        help="Puste = wszystkie sklepy.",
    )
    run_clicked = st.button("🔄 Odśwież przegląd", type="primary", use_container_width=True)
    st.caption(f"Sklepów w portfolio: **{len(core.property_map)}**")


# ─────────────────────────────────────────────────────────────
# LICZENIE
# ─────────────────────────────────────────────────────────────
def run_overview(stores: pd.DataFrame, lookback_days: int,
                  trend_metric: str, sigma_threshold: float) -> dict:
    end = core.yesterday
    start = end - timedelta(days=lookback_days - 1)

    trend = core.compute_trend_summary(stores, trend_metric, start, end)
    anomalies = core.detect_anomalies_for_stores(
        stores, core.OVERVIEW_METRICS, sigma_threshold=sigma_threshold
    )
    cart_anomalies = core.detect_cart_abandonment_anomalies(
        stores, sigma_threshold=sigma_threshold
    )

    return {
        "period":         {"start": str(start), "end": str(end)},
        "trend":          trend,
        "trend_metric":   trend_metric,
        "anomalies":      anomalies,
        "cart_anomalies": cart_anomalies,
    }


if "overview_result" not in st.session_state:
    st.session_state["overview_result"] = None

if run_clicked:
    stores_to_check = core.property_map
    if brand_filter:
        stores_to_check = stores_to_check[stores_to_check["Brand"].isin(brand_filter)]
    if stores_to_check.empty:
        st.error("Brak sklepów dla wybranych filtrów.")
    else:
        with st.spinner("Liczę trendy i szukam anomalii…"):
            st.session_state["overview_result"] = run_overview(
                stores_to_check, lookback_days, trend_metric, sigma_threshold
            )

result = st.session_state["overview_result"]

if result is None:
    st.info("Kliknij **🔄 Odśwież przegląd** w panelu bocznym, aby policzyć trendy i anomalie.")
    st.stop()

st.caption(f"Okres: {result['period']['start']} → {result['period']['end']}")

# ─────────────────────────────────────────────────────────────
# SEKCJA: TRENDY
# ─────────────────────────────────────────────────────────────
metric_label = core.METRIC_LABELS.get(result["trend_metric"], result["trend_metric"])
st.header(f"📈 Trendy — {metric_label}")

trend = result["trend"]

if not trend:
    st.info("Brak danych do wyliczenia trendów w wybranym okresie.")
else:
    rows = []
    for mpk, t in trend.items():
        pct = t["week_over_week_pct"]
        if pct is None:
            direction, sort_key = "❔ brak porównania", -1.0
        elif pct >= move_threshold:
            direction, sort_key = "🔺 rosnący", abs(pct)
        elif pct <= -move_threshold:
            direction, sort_key = "🔻 spadający", abs(pct)
        else:
            direction, sort_key = "➡️ stabilny", abs(pct)
        rows.append({
            "mpk": mpk, "brand": t["brand"], "direction": direction,
            "pct": pct, "latest_value": t["latest_value"], "sort_key": sort_key,
        })

    rows.sort(key=lambda r: r["sort_key"], reverse=True)

    n_up   = sum(1 for r in rows if r["direction"] == "🔺 rosnący")
    n_down = sum(1 for r in rows if r["direction"] == "🔻 spadający")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rosnące", n_up)
    c2.metric("Spadające", n_down)
    c3.metric("Stabilne / bez zmian", len(rows) - n_up - n_down)

    for r in rows:
        pct_label = f"{r['pct']:+.1f}%" if r["pct"] is not None else "brak danych"
        with st.expander(
            f"{r['direction']} — {r['mpk']} ({r['brand']}) · zmiana tydz./tydz. {pct_label} "
            f"· ostatnio {r['latest_value']}"
        ):
            df = trend[r["mpk"]]["df"].copy()
            df["date"] = df["date"].astype(str)
            fig = px.line(
                df, x="date", y=result["trend_metric"], markers=True,
                title=f"{metric_label} — {r['mpk']}",
                color_discrete_sequence=["#2ecc71"],
            )
            fig.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fafafa",
                xaxis=dict(showgrid=True, gridcolor="#2a2a3e"),
                yaxis=dict(showgrid=True, gridcolor="#2a2a3e"),
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SEKCJA: NOWE ANALIZY
# ─────────────────────────────────────────────────────────────
st.header("🔍 Nowe analizy")

insights = []
for a in result["anomalies"]["anomalies"]:
    for alert in a["alerts"]:
        insights.append({
            "MPK": a["MPK"], "Brand": a["Brand"], "metric": alert["metric"],
            "current": alert["current"], "hist_mean": alert["hist_mean"],
            "sigma_diff": alert["sigma_diff"], "direction": alert["direction"],
        })
insights.extend(result["cart_anomalies"])

if not insights:
    st.success("Brak nietypowych sygnałów w wybranym okresie i progu σ.")
else:
    insights.sort(key=lambda x: abs(x["sigma_diff"]), reverse=True)

    # (etykieta gdy "powyżej" średniej, etykieta gdy "poniżej" średniej)
    LABELS = {
        "totalRevenue":           ("💰 Pik przychodu",              "💸 Spadek przychodu"),
        "sessions":                ("📈 Wzrost ruchu",               "📉 Spadek ruchu"),
        "conversions":             ("🛒 Wzrost konwersji",           "⚠️ Spadek konwersji"),
        "sessionConversionRate":   ("🚀 CR w górę",                  "⚠️ CR w dół"),
        "bounceRate":              ("⚠️ Wsp. odbić w górę",          "✅ Wsp. odbić w dół"),
        "cart_abandonment_rate":   ("🛒⚠️ Więcej porzuconych koszyków", "✅ Mniej porzuconych koszyków"),
    }
    METRIC_NAMES = {**core.METRIC_LABELS, "cart_abandonment_rate": "Wsp. porzuconych koszyków"}

    st.warning(f"Znaleziono {len(insights)} nietypowych sygnałów (próg {sigma_threshold}σ).")
    for ins in insights:
        up_label, down_label = LABELS.get(ins["metric"], ("🔺 Wzrost", "🔻 Spadek"))
        label = up_label if ins["direction"] == "powyżej" else down_label
        metric_name = METRIC_NAMES.get(ins["metric"], ins["metric"])
        st.markdown(
            f"- **{label}** — {ins['MPK']} ({ins['Brand']}): "
            f"{metric_name} = {ins['current']} "
            f"(średnia z historii {ins['hist_mean']}, {ins['sigma_diff']}σ {ins['direction']})"
        )
