"""
GA4 AI Agent
============
Pełny agent konwersacyjny z tool calling, wykresami Plotly i pamięcią rozmowy.

Wymagania (requirements.txt):
    streamlit
    pandas
    google-analytics-data
    google-auth
    anthropic>=0.25.0
    plotly
    requests

secrets.toml (taki sam jak w eksporterze + anthropic):
    [app]
    password = "..."

    [gcp_service_account]
    # ... service account JSON

    [ga4_properties]
    # MPK = ["ga4_id", "Brand", "Currency"]

    [anthropic]
    api_key = "sk-ant-..."
"""

import json
from datetime import date, timedelta

import anthropic
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    RunReportRequest,
)
from google.oauth2 import service_account

# ─────────────────────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GA4 AI Agent",
    page_icon="🤖",
    layout="wide",
)

MONITORED_METRICS = ["sessions", "totalRevenue", "conversions", "bounceRate"]
METRIC_LABELS = {
    "sessions":     "Sesje",
    "totalRevenue": "Przychód",
    "conversions":  "Konwersje",
    "bounceRate":   "Wsp. odbić",
}

yesterday = date.today() - timedelta(days=1)

# ─────────────────────────────────────────────────────────────
# LOGOWANIE
# ─────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.title("🔐 GA4 AI Agent")
    pwd = st.text_input("Hasło:", type="password")
    if st.button("Zaloguj", use_container_width=True):
        if pwd == st.secrets["app"]["password"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ Błędne hasło!")
    st.stop()

# ─────────────────────────────────────────────────────────────
# KLIENTY
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_ga4_client():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/analytics.readonly"],
    )
    return BetaAnalyticsDataClient(credentials=creds)

@st.cache_resource
def get_ai_client():
    return anthropic.Anthropic(api_key=st.secrets["anthropic"]["api_key"])

ga4 = get_ga4_client()
ai  = get_ai_client()

# ─────────────────────────────────────────────────────────────
# MAPOWANIE SKLEPÓW
# ─────────────────────────────────────────────────────────────
property_map = pd.DataFrame([
    {
        "MPK":      mpk,
        "ID_GA4":   int(vals[0]),
        "Brand":    vals[1],
        "Currency": vals[2] if len(vals) > 2 else "PLN",
    }
    for mpk, vals in st.secrets["ga4_properties"].items()
])

MPK_INDEX    = {row["MPK"]:    row for _, row in property_map.iterrows()}
BRAND_INDEX  = {}
for _, row in property_map.iterrows():
    BRAND_INDEX.setdefault(row["Brand"], []).append(row)

# ─────────────────────────────────────────────────────────────
# KURSY NBP
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_nbp_rates():
    try:
        r = requests.get("https://api.nbp.pl/api/exchangerates/tables/A/?format=json", timeout=10)
        r.raise_for_status()
        rates = {"PLN": 1.0}
        for item in r.json()[0]["rates"]:
            rates[item["code"]] = item["mid"]
        return rates
    except Exception:
        return {"PLN": 1.0}

# ─────────────────────────────────────────────────────────────
# GA4 HELPERS
# ─────────────────────────────────────────────────────────────
def _parse_date(d: str) -> date:
    """Parsuje 'YYYY-MM-DD' lub relatywne 'NdaysAgo'."""
    if d == "yesterday":
        return yesterday
    if d == "today":
        return date.today()
    if d.endswith("daysAgo"):
        return yesterday - timedelta(days=int(d.replace("daysAgo", "")) - 1)
    return date.fromisoformat(d)


def _resolve_stores(mpks: list[str] | None, brands: list[str] | None) -> pd.DataFrame:
    """Zwraca przefiltrowany property_map."""
    df = property_map.copy()
    if mpks:
        df = df[df["MPK"].isin(mpks)]
    if brands:
        df = df[df["Brand"].isin(brands)]
    return df


def _fetch_aggregate(property_id: int, metrics: list[str],
                     start: date, end: date) -> dict:
    try:
        req = RunReportRequest(
            property=f"properties/{property_id}",
            metrics=[Metric(name=m) for m in metrics],
            date_ranges=[DateRange(start_date=str(start), end_date=str(end))],
        )
        resp = ga4.run_report(req)
        if not resp.rows:
            return {m: None for m in metrics}
        row = resp.rows[0]
        return {metrics[i]: float(mv.value) for i, mv in enumerate(row.metric_values)}
    except Exception as e:
        return {"error": str(e)}


def _fetch_daily(property_id: int, metrics: list[str],
                 start: date, end: date,
                 dimensions_extra: list[str] | None = None) -> pd.DataFrame:
    dims = ["date"] + (dimensions_extra or [])
    try:
        req = RunReportRequest(
            property=f"properties/{property_id}",
            dimensions=[Dimension(name=d) for d in dims],
            metrics=[Metric(name=m) for m in metrics],
            date_ranges=[DateRange(start_date=str(start), end_date=str(end))],
        )
        resp = ga4.run_report(req)
        rows = []
        for row in resp.rows:
            r = {dims[i]: dv.value for i, dv in enumerate(row.dimension_values)}
            for i, mv in enumerate(row.metric_values):
                r[metrics[i]] = float(mv.value)
            rows.append(r)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.date
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


# ─────────────────────────────────────────────────────────────
# NARZĘDZIA AGENTA (TOOL DEFINITIONS)
# ─────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "list_stores",
        "description": (
            "Zwraca listę wszystkich dostępnych sklepów (MPK, Brand, Currency). "
            "Użyj gdy użytkownik pyta o sklepy, chce wybrać sklep lub nie podał MPK."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "brand_filter": {
                    "type": "string",
                    "description": "Opcjonalnie filtruj po nazwie brandu (np. 'Nike').",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_metrics",
        "description": (
            "Pobiera zagregowane metryki GA4 dla wybranych sklepów i okresu czasu. "
            "Zwraca wartości bieżące oraz (opcjonalnie) porównanie do poprzedniego okresu. "
            "Użyj gdy pytanie dotyczy konkretnych liczb, wyników, przychodów itp."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mpks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista kodów MPK sklepów. Puste = wszystkie sklepy.",
                },
                "brands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filtruj po brandzie zamiast MPK (alternatywa).",
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": MONITORED_METRICS,
                    },
                    "description": "Metryki do pobrania. Domyślnie wszystkie.",
                },
                "start_date": {
                    "type": "string",
                    "description": "Data od, format YYYY-MM-DD lub '7daysAgo', 'yesterday'.",
                },
                "end_date": {
                    "type": "string",
                    "description": "Data do, format YYYY-MM-DD lub 'yesterday'.",
                },
                "compare_previous": {
                    "type": "boolean",
                    "description": "Czy dołączyć porównanie do poprzedniego okresu tej samej długości.",
                },
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "get_trend",
        "description": (
            "Pobiera dzienne dane GA4 dla wybranego sklepu/sklepów i metryki — "
            "do analizy trendu, wykrywania anomalii, sezonowości. "
            "Użyj gdy pytanie dotyczy trendu, historii, wykresu, zmian w czasie."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mpks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista MPK. Puste = wszystkie sklepy (uwaga: może być wolne).",
                },
                "brands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filtruj po brandzie.",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string", "enum": MONITORED_METRICS},
                    "description": "Metryki do pobrania.",
                },
                "start_date": {"type": "string", "description": "Data od."},
                "end_date":   {"type": "string", "description": "Data do."},
            },
            "required": ["metrics", "start_date", "end_date"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": (
            "Wykrywa anomalie statystyczne (odchylenia >N sigma od średniej 30-dniowej) "
            "dla wybranych sklepów i metryk. "
            "Użyj gdy pytanie dotyczy anomalii, problemów, spadków, alertów."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mpks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista MPK. Puste = wszystkie.",
                },
                "brands": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string", "enum": MONITORED_METRICS},
                    "description": "Metryki do sprawdzenia.",
                },
                "reference_date": {
                    "type": "string",
                    "description": "Dzień względem którego liczymy historię. Domyślnie yesterday.",
                },
                "sigma_threshold": {
                    "type": "number",
                    "description": "Próg odchylenia standardowego. Domyślnie 2.0.",
                },
            },
            "required": ["metrics"],
        },
    },
    {
        "name": "plot_trend",
        "description": (
            "Rysuje wykres liniowy trendu dla podanych danych. "
            "Użyj PO pobraniu danych przez get_trend, przekazując te same parametry. "
            "Wykres pojawi się bezpośrednio w czacie."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mpks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "brands": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "metric": {
                    "type": "string",
                    "enum": MONITORED_METRICS,
                    "description": "Jedna metryka do wykresu.",
                },
                "start_date": {"type": "string"},
                "end_date":   {"type": "string"},
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "bar", "area"],
                    "description": "Typ wykresu. Domyślnie line.",
                },
            },
            "required": ["metric", "start_date", "end_date"],
        },
    },
    {
        "name": "compare_stores",
        "description": (
            "Porównuje wiele sklepów względem siebie dla wybranych metryk i okresu. "
            "Zwraca ranking oraz wykres słupkowy. "
            "Użyj gdy pytanie dotyczy porównania sklepów, rankingu, najlepszych/najgorszych."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mpks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Puste = wszystkie sklepy.",
                },
                "brands": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string", "enum": MONITORED_METRICS},
                },
                "start_date": {"type": "string"},
                "end_date":   {"type": "string"},
                "sort_by": {
                    "type": "string",
                    "enum": MONITORED_METRICS,
                    "description": "Po jakiej metryce sortować ranking.",
                },
            },
            "required": ["metrics", "start_date", "end_date"],
        },
    },
]

# ─────────────────────────────────────────────────────────────
# IMPLEMENTACJE NARZĘDZI
# ─────────────────────────────────────────────────────────────
def tool_list_stores(brand_filter: str | None = None) -> dict:
    df = property_map.copy()
    if brand_filter:
        df = df[df["Brand"].str.contains(brand_filter, case=False, na=False)]
    return {
        "stores": df[["MPK", "Brand", "Currency"]].to_dict(orient="records"),
        "total": len(df),
    }


def tool_get_metrics(
    start_date: str,
    end_date: str,
    mpks: list[str] | None = None,
    brands: list[str] | None = None,
    metrics: list[str] | None = None,
    compare_previous: bool = False,
) -> dict:
    stores = _resolve_stores(mpks, brands)
    if stores.empty:
        return {"error": "Nie znaleziono sklepów dla podanych filtrów."}

    mets = metrics or MONITORED_METRICS
    s = _parse_date(start_date)
    e = _parse_date(end_date)
    delta = (e - s).days + 1
    cmp_e = s - timedelta(days=1)
    cmp_s = cmp_e - timedelta(days=delta - 1)

    results = []
    for _, row in stores.iterrows():
        current = _fetch_aggregate(row["ID_GA4"], mets, s, e)
        entry = {"MPK": row["MPK"], "Brand": row["Brand"], "current": current}
        if compare_previous:
            prev = _fetch_aggregate(row["ID_GA4"], mets, cmp_s, cmp_e)
            entry["previous"] = prev
            entry["change"] = {
                m: round((current[m] - prev[m]) / prev[m], 4)
                if prev.get(m) and prev[m] != 0 else None
                for m in mets
                if current.get(m) is not None
            }
        results.append(entry)

    return {
        "period": {"start": str(s), "end": str(e)},
        "comparison_period": {"start": str(cmp_s), "end": str(cmp_e)} if compare_previous else None,
        "results": results,
    }


def tool_get_trend(
    metrics: list[str],
    start_date: str,
    end_date: str,
    mpks: list[str] | None = None,
    brands: list[str] | None = None,
) -> dict:
    stores = _resolve_stores(mpks, brands)
    if stores.empty:
        return {"error": "Nie znaleziono sklepów."}

    s = _parse_date(start_date)
    e = _parse_date(end_date)
    all_dfs = []

    for _, row in stores.iterrows():
        df = _fetch_daily(row["ID_GA4"], metrics, s, e)
        if not df.empty and "error" not in df.columns:
            df["MPK"]   = row["MPK"]
            df["Brand"] = row["Brand"]
            all_dfs.append(df)

    if not all_dfs:
        return {"error": "Brak danych dla podanego okresu."}

    combined = pd.concat(all_dfs, ignore_index=True)
    # Zwróć skrócone dane (max 200 wierszy) żeby nie zapychać kontekstu
    sample = combined.head(200)
    return {
        "period": {"start": str(s), "end": str(e)},
        "rows": len(combined),
        "data": sample.to_dict(orient="records"),
        "columns": list(sample.columns),
    }


def tool_detect_anomalies(
    metrics: list[str],
    mpks: list[str] | None = None,
    brands: list[str] | None = None,
    reference_date: str = "yesterday",
    sigma_threshold: float = 2.0,
) -> dict:
    stores = _resolve_stores(mpks, brands)
    if stores.empty:
        return {"error": "Nie znaleziono sklepów."}

    ref = _parse_date(reference_date)
    hist_end   = ref - timedelta(days=1)
    hist_start = hist_end - timedelta(days=29)

    anomalies = []
    summaries = []

    for _, row in stores.iterrows():
        pid = row["ID_GA4"]
        # Dzisiejsze/bieżące wartości
        cur = _fetch_aggregate(pid, metrics, ref, ref)
        # Historia 30 dni
        hist_df = _fetch_daily(pid, metrics, hist_start, hist_end)

        store_anomalies = []
        store_summary = {"MPK": row["MPK"], "Brand": row["Brand"], "metrics": {}}

        for m in metrics:
            val = cur.get(m)
            if val is None:
                continue

            hist_mean = hist_std = None
            if not hist_df.empty and m in hist_df.columns:
                vals = hist_df[m].dropna()
                if len(vals) > 1:
                    hist_mean = float(vals.mean())
                    hist_std  = float(vals.std())

            is_anomaly = False
            sigma_diff = None
            if hist_mean is not None and hist_std and hist_std > 0:
                sigma_diff = (val - hist_mean) / hist_std
                is_anomaly = abs(sigma_diff) > sigma_threshold

            store_summary["metrics"][m] = {
                "current":   round(val, 2),
                "hist_mean": round(hist_mean, 2) if hist_mean else None,
                "hist_std":  round(hist_std, 2)  if hist_std  else None,
                "sigma_diff": round(sigma_diff, 2) if sigma_diff else None,
                "is_anomaly": is_anomaly,
            }

            if is_anomaly:
                store_anomalies.append({
                    "metric":     m,
                    "current":    round(val, 2),
                    "hist_mean":  round(hist_mean, 2),
                    "sigma_diff": round(sigma_diff, 2),
                    "direction":  "powyżej" if sigma_diff > 0 else "poniżej",
                })

        summaries.append(store_summary)
        if store_anomalies:
            anomalies.append({
                "MPK":    row["MPK"],
                "Brand":  row["Brand"],
                "alerts": store_anomalies,
            })

    return {
        "reference_date":  str(ref),
        "history_period":  f"{hist_start} → {hist_end}",
        "sigma_threshold": sigma_threshold,
        "anomalies_found": len(anomalies),
        "anomalies":       anomalies,
        "all_stores":      summaries,
    }


# Przechowujemy wykresy do renderowania
_PENDING_CHARTS: list = []

def tool_plot_trend(
    metric: str,
    start_date: str,
    end_date: str,
    mpks: list[str] | None = None,
    brands: list[str] | None = None,
    chart_type: str = "line",
) -> dict:
    stores = _resolve_stores(mpks, brands)
    if stores.empty:
        return {"error": "Nie znaleziono sklepów."}

    s = _parse_date(start_date)
    e = _parse_date(end_date)
    all_dfs = []

    for _, row in stores.iterrows():
        df = _fetch_daily(row["ID_GA4"], [metric], s, e)
        if not df.empty and "error" not in df.columns:
            df["Sklep"] = f"{row['MPK']} – {row['Brand']}"
            all_dfs.append(df)

    if not all_dfs:
        return {"error": "Brak danych dla wykresu."}

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["date"] = combined["date"].astype(str)

    title = f"{METRIC_LABELS.get(metric, metric)} | {s} → {e}"

    if chart_type == "bar":
        fig = px.bar(
            combined, x="date", y=metric, color="Sklep",
            title=title, barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    elif chart_type == "area":
        fig = px.area(
            combined, x="date", y=metric, color="Sklep",
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    else:
        fig = px.line(
            combined, x="date", y=metric, color="Sklep",
            title=title, markers=True,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        xaxis=dict(showgrid=True, gridcolor="#2a2a3e"),
        yaxis=dict(showgrid=True, gridcolor="#2a2a3e"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=40),
    )

    _PENDING_CHARTS.append(fig)
    return {"status": "Wykres wygenerowany i zostanie wyświetlony w czacie.", "rows": len(combined)}


def tool_compare_stores(
    metrics: list[str],
    start_date: str,
    end_date: str,
    mpks: list[str] | None = None,
    brands: list[str] | None = None,
    sort_by: str | None = None,
) -> dict:
    stores = _resolve_stores(mpks, brands)
    if stores.empty:
        return {"error": "Nie znaleziono sklepów."}

    s = _parse_date(start_date)
    e = _parse_date(end_date)

    rows = []
    for _, row in stores.iterrows():
        cur = _fetch_aggregate(row["ID_GA4"], metrics, s, e)
        entry = {"MPK": row["MPK"], "Brand": row["Brand"]}
        entry.update({m: round(v, 2) if v else 0 for m, v in cur.items() if m in metrics})
        rows.append(entry)

    if not rows:
        return {"error": "Brak danych."}

    sort_col = sort_by or metrics[0]
    rows.sort(key=lambda x: x.get(sort_col, 0), reverse=True)

    # Wykres słupkowy rankingu
    df_rank = pd.DataFrame(rows)
    df_rank["Sklep"] = df_rank["MPK"] + " – " + df_rank["Brand"]

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, m in enumerate(metrics):
        if m in df_rank.columns:
            fig.add_trace(go.Bar(
                name=METRIC_LABELS.get(m, m),
                x=df_rank["Sklep"],
                y=df_rank[m],
                marker_color=colors[i % len(colors)],
            ))

    fig.update_layout(
        title=f"Porównanie sklepów | {s} → {e}",
        barmode="group",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#2a2a3e"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=80),
    )
    fig.update_xaxes(tickangle=-30)
    _PENDING_CHARTS.append(fig)

    return {
        "period": {"start": str(s), "end": str(e)},
        "ranking": rows,
        "chart": "Wykres porównawczy zostanie wyświetlony.",
    }


# ─────────────────────────────────────────────────────────────
# DISPATCHER – wywołuje właściwe narzędzie
# ─────────────────────────────────────────────────────────────
def dispatch_tool(name: str, inputs: dict) -> str:
    try:
        if name == "list_stores":
            result = tool_list_stores(**inputs)
        elif name == "get_metrics":
            result = tool_get_metrics(**inputs)
        elif name == "get_trend":
            result = tool_get_trend(**inputs)
        elif name == "detect_anomalies":
            result = tool_detect_anomalies(**inputs)
        elif name == "plot_trend":
            result = tool_plot_trend(**inputs)
        elif name == "compare_stores":
            result = tool_compare_stores(**inputs)
        else:
            result = {"error": f"Nieznane narzędzie: {name}"}
    except Exception as e:
        result = {"error": str(e)}

    return json.dumps(result, ensure_ascii=False, default=str)


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
STORE_LIST_SHORT = ", ".join(
    f"{r['MPK']} ({r['Brand']})" for _, r in property_map.head(20).iterrows()
)

SYSTEM_PROMPT = f"""Jesteś GA4 AI Agentem — ekspertem analityki e-commerce analizującym dane z Google Analytics 4.

## Twoje możliwości
- Pobierasz metryki GA4 w czasie rzeczywistym używając narzędzi
- Wykrywasz anomalie statystyczne i trendy
- Generujesz wykresy które pojawiają się bezpośrednio w czacie
- Porównujesz sklepy i tworzysz rankingi
- Interpretujesz dane biznesowo i dajesz konkretne zalecenia

## Dostępne sklepy ({len(property_map)} łącznie)
{STORE_LIST_SHORT}{"..." if len(property_map) > 20 else ""}

## Dostępne metryki
- sessions – liczba sesji
- totalRevenue – przychód (w walucie sklepu)
- conversions – liczba konwersji
- bounceRate – współczynnik odrzuceń (wyższy = gorszy)

## Zasady działania
1. ZAWSZE używaj narzędzi do pobierania danych — nigdy nie zmyślaj liczb
2. Gdy użytkownik pyta o wykres, NAJPIERW wywołaj plot_trend lub compare_stores
3. Przy porównaniach zawsze dodaj kontekst (czy to dobry/zły wynik i dlaczego)
4. Jeśli pytanie jest niejasne — zapytaj o MPK lub zakres dat
5. Odpowiadaj po polsku, zwięźle i rzeczowo
6. Dla anomalii zawsze sugeruj możliwe przyczyny i kroki naprawcze
7. Dzisiejsze dane mogą być niekompletne — informuj o tym gdy użytkownik pyta o "dziś"

## Format odpowiedzi
- Używaj emoji sparingowo dla czytelności
- Liczby formatuj z separatorami (1 234, nie 1234)
- Zmiany podawaj jako % i wartość bezwzględną
- Rankingi jako numerowaną listę
"""

# ─────────────────────────────────────────────────────────────
# AGENTIC LOOP
# ─────────────────────────────────────────────────────────────
def run_agent(user_message: str, history: list[dict]) -> tuple[str, list]:
    """
    Uruchamia agenta z tool calling.
    Zwraca (final_text, updated_history).
    """
    global _PENDING_CHARTS
    _PENDING_CHARTS = []

    messages = history + [{"role": "user", "content": user_message}]

    tool_calls_log = []  # do wyświetlenia w UI

    for _ in range(10):  # max 10 iteracji tool calling
        response = ai.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Dodaj odpowiedź asystenta do historii
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Wyodrębnij tekst końcowej odpowiedzi
            final_text = " ".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return final_text, messages, tool_calls_log

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls_log.append(f"🔧 `{block.name}` — {json.dumps(block.input, ensure_ascii=False)[:120]}")
                    result_str = dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result_str,
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

        break  # nieoczekiwany stop_reason

    return "Przepraszam, coś poszło nie tak w pętli agenta.", messages, tool_calls_log


# ─────────────────────────────────────────────────────────────
# UI – SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 GA4 AI Agent")
    st.markdown("---")

    st.subheader("📊 Portfolio")
    st.metric("Sklepów", len(property_map))
    brands_count = property_map["Brand"].nunique()
    st.metric("Brandów", brands_count)
    st.caption(f"Dane do: **{yesterday}**")

    st.markdown("---")
    st.subheader("💡 Przykładowe pytania")
    examples = [
        "Jakie były przychody wszystkich sklepów w ostatnich 7 dniach?",
        "Pokaż trend sesji dla sklepu X w ostatnim miesiącu",
        "Które sklepy mają anomalie w konwersjach?",
        "Porównaj top 5 sklepów według przychodu",
        "Dlaczego sklep X ma niski przychód w tym tygodniu?",
        "Narysuj wykres bounceRate dla brandów Y i Z",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["quick_input"] = ex

    st.markdown("---")
    if st.button("🗑️ Wyczyść historię czatu", use_container_width=True):
        st.session_state["chat_history"]  = []
        st.session_state["display_history"] = []
        st.rerun()

# ─────────────────────────────────────────────────────────────
# UI – GŁÓWNY CZAT
# ─────────────────────────────────────────────────────────────
st.title("🤖 GA4 AI Agent")
st.caption("Zadaj pytanie o dane ze swoich sklepów — agent sam pobierze odpowiednie dane z GA4.")

# Inicjalizacja stanu
if "chat_history" not in st.session_state:
    st.session_state["chat_history"]    = []   # historia dla API (role/content)
    st.session_state["display_history"] = []   # historia do wyświetlenia w UI

# Wiadomość powitalna
if not st.session_state["display_history"]:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            f"Cześć! Jestem Twoim agentem GA4. Mam dostęp do **{len(property_map)} sklepów** "
            f"z {brands_count} brandów.\n\n"
            "Możesz mnie zapytać o:\n"
            "- 📈 Trendy i wykresy metryk\n"
            "- 🔍 Anomalie i spadki\n"
            "- 🏆 Rankingi i porównania sklepów\n"
            "- 💡 Interpretację i zalecenia\n\n"
            """Spróbuj: *„Które sklepy mają największy spadek sesji w tym tygodniu?”*"""
        )

# Wyświetl historię
for entry in st.session_state["display_history"]:
    with st.chat_message(entry["role"], avatar="👤" if entry["role"] == "user" else "🤖"):
        st.markdown(entry["content"])
        # Wykresy dołączone do wiadomości asystenta
        for fig in entry.get("charts", []):
            st.plotly_chart(fig, use_container_width=True)
        # Log narzędzi (zwinięty)
        if entry.get("tool_calls"):
            with st.expander(f"🔧 Wywołane narzędzia ({len(entry['tool_calls'])})", expanded=False):
                for tc in entry["tool_calls"]:
                    st.code(tc, language=None)

# Input (obsługa quick_input z sidebaru)
quick = st.session_state.pop("quick_input", None)
user_input = st.chat_input("Zapytaj o swoje sklepy…") or quick

if user_input:
    # Wyświetl wiadomość użytkownika
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    st.session_state["display_history"].append({"role": "user", "content": user_input})

    # Uruchom agenta
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Agent analizuje…"):
            reply, updated_history, tool_calls = run_agent(
                user_input,
                st.session_state["chat_history"],
            )

        # Pobierz wykresy z globalnej listy
        charts = list(_PENDING_CHARTS)
        _PENDING_CHARTS = []

        st.markdown(reply)
        for fig in charts:
            st.plotly_chart(fig, use_container_width=True)

        if tool_calls:
            with st.expander(f"🔧 Wywołane narzędzia ({len(tool_calls)})", expanded=False):
                for tc in tool_calls:
                    st.code(tc, language=None)

    # Zapisz do historii (bez wiadomości użytkownika która już jest w updated_history)
    st.session_state["chat_history"] = updated_history
    st.session_state["display_history"].append({
        "role":       "assistant",
        "content":    reply,
        "charts":     charts,
        "tool_calls": tool_calls,
    })
