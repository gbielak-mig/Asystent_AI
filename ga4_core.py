"""
GA4 Core
========
Wspólne klienci, mapowanie sklepów i funkcje pobierania danych z GA4,
używane zarówno przez czat agenta (Agent_AI.py) jak i stronę audytu
(pages/1_Audyt.py).
"""

from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st
from google.analytics.admin_v1beta import AnalyticsAdminServiceClient
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    RunReportRequest,
)
from google.oauth2 import service_account

MONITORED_METRICS = ["sessions", "totalRevenue", "conversions", "bounceRate"]
METRIC_LABELS = {
    "sessions":     "Sesje",
    "totalRevenue": "Przychód",
    "conversions":  "Konwersje",
    "bounceRate":   "Wsp. odbić",
}

yesterday = date.today() - timedelta(days=1)


# ─────────────────────────────────────────────────────────────
# AUTORYZACJA
# ─────────────────────────────────────────────────────────────
def require_auth() -> None:
    """Wyświetla ekran logowania; zatrzymuje renderowanie strony, jeśli brak sesji."""
    if st.session_state.get("authenticated"):
        return
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
# KLIENCI
# ─────────────────────────────────────────────────────────────
def _service_account_creds(scopes: list[str]):
    return service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes,
    )


@st.cache_resource
def get_ga4_client() -> BetaAnalyticsDataClient:
    creds = _service_account_creds(["https://www.googleapis.com/auth/analytics.readonly"])
    return BetaAnalyticsDataClient(credentials=creds)


@st.cache_resource
def get_ga4_admin_client() -> AnalyticsAdminServiceClient:
    creds = _service_account_creds(["https://www.googleapis.com/auth/analytics.readonly"])
    return AnalyticsAdminServiceClient(credentials=creds)


@st.cache_resource
def get_ai_client():
    import anthropic
    return anthropic.Anthropic(api_key=st.secrets["anthropic"]["api_key"])


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

MPK_INDEX   = {row["MPK"]: row for _, row in property_map.iterrows()}
BRAND_INDEX = {}
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
# GA4 DATA API HELPERS
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
        resp = get_ga4_client().run_report(req)
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
        resp = get_ga4_client().run_report(req)
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


def fetch_event_counts(property_id: int, start: date, end: date) -> dict:
    """Zwraca {event_name: liczba_zdarzeń} dla property w okresie,
    albo {"__error__": komunikat} gdy zapytanie się nie powiodło."""
    try:
        req = RunReportRequest(
            property=f"properties/{property_id}",
            dimensions=[Dimension(name="eventName")],
            metrics=[Metric(name="eventCount")],
            date_ranges=[DateRange(start_date=str(start), end_date=str(end))],
        )
        resp = get_ga4_client().run_report(req)
        return {
            row.dimension_values[0].value: int(float(row.metric_values[0].value))
            for row in resp.rows
        }
    except Exception as e:
        return {"__error__": str(e)}


def fetch_custom_dimension_activity(property_id: int, parameter_name: str,
                                     scope: str, start: date, end: date) -> int | None:
    """Zwraca liczbę zdarzeń z niepustą wartością danego custom dimension
    w okresie, albo None gdy zapytanie się nie powiodło."""
    prefix = "customUser" if scope == "USER" else "customEvent"
    try:
        req = RunReportRequest(
            property=f"properties/{property_id}",
            dimensions=[Dimension(name=f"{prefix}:{parameter_name}")],
            metrics=[Metric(name="eventCount")],
            date_ranges=[DateRange(start_date=str(start), end_date=str(end))],
        )
        resp = get_ga4_client().run_report(req)
        total = 0
        for row in resp.rows:
            val = row.dimension_values[0].value
            if val and val != "(not set)":
                total += int(float(row.metric_values[0].value))
        return total
    except Exception:
        return None


def fetch_admin_config(property_id: int) -> dict:
    """Pobiera z GA4 Admin API skonfigurowane custom dimensions, custom metrics
    i key events (eventy konwersji) dla danej property."""
    admin = get_ga4_admin_client()
    parent = f"properties/{property_id}"
    result = {"custom_dimensions": [], "custom_metrics": [], "key_events": [], "error": None}
    try:
        result["custom_dimensions"] = [
            {
                "parameter_name": cd.parameter_name,
                "display_name": cd.display_name,
                "scope": cd.scope.name,
            }
            for cd in admin.list_custom_dimensions(parent=parent)
        ]
        result["custom_metrics"] = [
            {
                "parameter_name": cm.parameter_name,
                "display_name": cm.display_name,
                "scope": cm.scope.name,
            }
            for cm in admin.list_custom_metrics(parent=parent)
        ]
        try:
            result["key_events"] = [ke.event_name for ke in admin.list_key_events(parent=parent)]
        except Exception:
            result["key_events"] = [ce.event_name for ce in admin.list_conversion_events(parent=parent)]
    except Exception as e:
        result["error"] = str(e)
    return result


# ─────────────────────────────────────────────────────────────
# WYKRYWANIE ANOMALII
# ─────────────────────────────────────────────────────────────
def detect_anomalies_for_stores(
    stores: pd.DataFrame,
    metrics: list[str],
    reference_date: str = "yesterday",
    sigma_threshold: float = 2.0,
) -> dict:
    """Wykrywa odchylenia >sigma_threshold od 30-dniowej średniej dla podanych
    sklepów i metryk. `stores` to przefiltrowany (lub pełny) property_map."""
    ref = _parse_date(reference_date)
    hist_end   = ref - timedelta(days=1)
    hist_start = hist_end - timedelta(days=29)

    anomalies = []
    summaries = []

    for _, row in stores.iterrows():
        pid = row["ID_GA4"]
        cur = _fetch_aggregate(pid, metrics, ref, ref)
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
