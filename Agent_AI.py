"""
GA4 AI Agent
============
Pełny agent konwersacyjny z tool calling, wykresami Plotly i pamięcią rozmowy.

Wymagania (requirements.txt):
    streamlit
    pandas
    google-analytics-data
    google-analytics-admin
    google-auth
    groq
    plotly
    requests

secrets.toml:
    [app]
    password = "..."

    [gcp_service_account]
    # ... service account JSON

    [ga4_properties]
    # MPK = ["ga4_id", "Brand", "Currency"]

    [groq]
    api_key = "gsk_..."
    # model = "openai/gpt-oss-20b"   # opcjonalnie, nadpisuje domyślny model
"""

import json
from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ga4_core import (
    METRIC_LABELS,
    MONITORED_METRICS,
    _fetch_aggregate,
    _fetch_daily,
    _parse_date,
    _resolve_stores,
    detect_anomalies_for_stores,
    get_ai_client,
    get_ai_model,
    get_ga4_client,
    property_map,
    require_auth,
    yesterday,
)

# ─────────────────────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GA4 AI Agent",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# LOGOWANIE
# ─────────────────────────────────────────────────────────────
require_auth()

# ─────────────────────────────────────────────────────────────
# KLIENTY
# ─────────────────────────────────────────────────────────────
ga4 = get_ga4_client()
ai  = get_ai_client()


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
            "Zwraca gotowo policzone statystyki (peak_date, week_over_week_pct, mean) — "
            "UŻYWAJ TYCH LICZB zamiast samodzielnie liczyć maksima/trendy z surowych danych. "
            "Gdy chart=true (domyślnie) i podana jedna metryka, automatycznie generuje też "
            "wykres — nie wywołuj osobno żadnego narzędzia do rysowania. "
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
                "chart": {
                    "type": "boolean",
                    "description": "Czy dołączyć wykres (działa tylko dla 1 metryki). Domyślnie true.",
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "bar", "area"],
                    "description": "Typ wykresu. Domyślnie line.",
                },
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
    chart: bool = True,
    chart_type: str = "line",
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

    # Statystyki policzone po stronie kodu (nie przez model) — peak, ostatnia
    # wartość, zmiana tydzień-do-tygodnia. Małe modele czatowe często się mylą
    # przy ręcznym czytaniu maksimów/trendów z surowej tabeli.
    summary = {}
    for m in metrics:
        per_store = {}
        for mpk, g in combined.groupby("MPK"):
            vals = g[["date", m]].dropna().sort_values("date")
            if vals.empty:
                continue
            peak_row = vals.loc[vals[m].idxmax()]
            latest_row = vals.iloc[-1]
            wow_pct = None
            prev_week = vals[vals["date"] <= (e - timedelta(days=7))]
            if not prev_week.empty:
                prev_val = prev_week.iloc[-1][m]
                if prev_val:
                    wow_pct = round((latest_row[m] - prev_val) / prev_val * 100, 1)
            per_store[mpk] = {
                "mean":               round(vals[m].mean(), 2),
                "peak_date":          str(peak_row["date"]),
                "peak_value":         round(peak_row[m], 2),
                "latest_date":        str(latest_row["date"]),
                "latest_value":       round(latest_row[m], 2),
                "week_over_week_pct": wow_pct,
            }
        summary[m] = per_store

    result = {
        "period": {"start": str(s), "end": str(e)},
        "rows": len(combined),
        "summary": summary,
        "columns": list(combined.columns),
    }

    if chart and len(metrics) == 1:
        result["chart"] = _queue_trend_chart(combined, metrics[0], s, e, chart_type)
        # Wykres + "summary" wystarczą do odpowiedzi — surowe wiersze pominięte,
        # żeby nie zapychać kontekstu i nie zjadać limitu tokenów/minutę (miały
        # one udział w realnym 429 na darmowym tierze Groq).
        result["data"] = "pominięte — użyj pól 'summary' i informacji o wykresie"
    else:
        # Bez wykresu: skrócone dane (max 60 wierszy), żeby nie zapychać kontekstu
        result["data"] = combined.head(60).to_dict(orient="records")

    return result


# Przechowujemy wykresy do renderowania
_PENDING_CHARTS: list = []

def _queue_trend_chart(combined: pd.DataFrame, metric: str, s, e, chart_type: str) -> str:
    plot_df = combined.copy()
    plot_df["date"] = plot_df["date"].astype(str)
    plot_df["Sklep"] = plot_df["MPK"] + " – " + plot_df["Brand"]

    title = f"{METRIC_LABELS.get(metric, metric)} | {s} → {e}"

    if chart_type == "bar":
        fig = px.bar(
            plot_df, x="date", y=metric, color="Sklep",
            title=title, barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    elif chart_type == "area":
        fig = px.area(
            plot_df, x="date", y=metric, color="Sklep",
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    else:
        fig = px.line(
            plot_df, x="date", y=metric, color="Sklep",
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
    return "Wykres wygenerowany i zostanie wyświetlony w czacie."


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
    return detect_anomalies_for_stores(stores, metrics, reference_date, sigma_threshold)


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
        elif name == "compare_stores":
            result = tool_compare_stores(**inputs)
        else:
            result = {"error": f"Nieznane narzędzie: {name}"}
    except Exception as e:
        result = {"error": str(e)}

    return json.dumps(result, ensure_ascii=False, default=str)


# Groq (OpenAI-compatible) format narzędzi: {"type": "function", "function": {...}}
GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOLS
]


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
2. Gdy użytkownik pyta o trend/wykres, wywołaj get_trend (domyślnie sam dołącza wykres)
   albo compare_stores przy porównaniach — NIE wywołuj do tego samego pytania dodatkowych
   narzędzi, to niepotrzebnie wydłuża odpowiedź
3. get_trend zwraca gotowo policzone pole "summary" (peak_date, peak_value, latest_value,
   week_over_week_pct, mean) per sklep i metryka — ZAWSZE cytuj te liczby, NIE licz
   samodzielnie maksimów/zmian % z surowych wierszy w polu "data" (łatwo się pomylić)
4. Przy porównaniach zawsze dodaj kontekst (czy to dobry/zły wynik i dlaczego)
5. Jeśli pytanie jest niejasne — zapytaj o MPK lub zakres dat
6. Odpowiadaj po polsku, zwięźle i rzeczowo
7. Dla anomalii zawsze sugeruj możliwe przyczyny i kroki naprawcze
8. Dzisiejsze dane mogą być niekompletne — informuj o tym gdy użytkownik pyta o "dziś"

## Format odpowiedzi
- Używaj emoji sparingowo dla czytelności
- Liczby formatuj z separatorami (1 234, nie 1234)
- Zmiany podawaj jako % i wartość bezwzględną
- Rankingi jako numerowaną listę
"""

# ─────────────────────────────────────────────────────────────
# AGENTIC LOOP
# ─────────────────────────────────────────────────────────────
def run_agent(user_message: str, history: list[dict]) -> tuple[str, list, list]:
    """
    Uruchamia agenta z tool calling (Groq, format OpenAI-compatible).
    Zwraca (final_text, updated_history, tool_calls_log).
    """
    global _PENDING_CHARTS
    _PENDING_CHARTS = []

    messages = history + [{"role": "user", "content": user_message}]

    tool_calls_log = []  # do wyświetlenia w UI

    for _ in range(10):  # max 10 iteracji tool calling
        response = ai.chat.completions.create(
            model=get_ai_model(),
            max_tokens=4096,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=GROQ_TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                inputs = json.loads(tc.function.arguments)
                tool_calls_log.append(f"🔧 `{tc.function.name}` — {json.dumps(inputs, ensure_ascii=False)[:120]}")
                result_str = dispatch_tool(tc.function.name, inputs)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_str,
                })
            continue

        # Brak wywołań narzędzi — to finalna odpowiedź
        messages.append({"role": "assistant", "content": msg.content})
        return msg.content or "", messages, tool_calls_log

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
