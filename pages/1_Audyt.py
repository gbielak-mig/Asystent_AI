"""
GA4 Audyt Wdrożenia
====================
Samodzielna strona audytu — sprawdza dla WSZYSTKICH sklepów naraz:
  1. konfigurację GA4 (custom dimensions/metrics, key events z Admin API),
  2. czy skonfigurowane parametry/eventy faktycznie zbierają dane,
  3. anomalie statystyczne w metrykach (odchylenia od 30-dniowej średniej),
  4. spójność wdrożenia eventów między sklepami tego samego brandu.

Wymaga w secrets.toml uprawnień service accounta do GA4 Admin API
(ten sam scope analytics.readonly wystarcza do odczytu).
"""

from datetime import timedelta

import pandas as pd
import streamlit as st

import ga4_core as core

st.set_page_config(page_title="GA4 Audyt", page_icon="🩺", layout="wide")
core.require_auth()

st.title("🩺 Audyt wdrożenia GA4")
st.caption(
    "Pełny audyt wszystkich sklepów naraz: konfiguracja, martwe parametry, "
    "anomalie i spójność wdrożenia między sklepami tego samego brandu."
)

# ─────────────────────────────────────────────────────────────
# USTAWIENIA
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Ustawienia audytu")
    lookback_days = st.slider("Okres sprawdzania aktywności (dni)", 7, 90, 30)
    sigma_threshold = st.slider("Próg anomalii (σ)", 1.0, 4.0, 2.0, step=0.5)
    brand_options = sorted(core.property_map["Brand"].unique())
    brand_filter = st.multiselect(
        "Ogranicz do brandów (opcjonalnie)", brand_options,
        help="Puste = audyt obejmuje wszystkie sklepy.",
    )
    run_clicked = st.button("▶️ Uruchom pełny audyt", type="primary", use_container_width=True)
    st.caption(f"Sklepów w portfolio: **{len(core.property_map)}**")


# ─────────────────────────────────────────────────────────────
# LOGIKA AUDYTU
# ─────────────────────────────────────────────────────────────
def run_full_audit(stores: pd.DataFrame, lookback_days: int, sigma_threshold: float) -> dict:
    end = core.yesterday
    start = end - timedelta(days=lookback_days - 1)

    config_rows = []
    dead_params = []
    store_events: dict[str, set] = {}

    total = len(stores)
    progress = st.progress(0.0, text="Rozpoczynanie audytu…")

    for i, (_, row) in enumerate(stores.iterrows()):
        pid = row["ID_GA4"]
        progress.progress(i / total, text=f"Sprawdzam {row['MPK']} ({row['Brand']})…")

        admin_cfg = core.fetch_admin_config(pid)
        event_counts = core.fetch_event_counts(pid, start, end)
        events_error = "__error__" in event_counts
        active_events = set(event_counts) - {"__error__"}
        store_events[row["MPK"]] = active_events

        if admin_cfg["error"]:
            config_rows.append({
                "MPK": row["MPK"], "Brand": row["Brand"], "Typ": "Admin API",
                "Nazwa": "—", "Zdarzenia w okresie": None,
                "Status": f"⚠️ Błąd Admin API: {admin_cfg['error'][:150]}",
            })
            continue

        # Key events (eventy konwersji): skonfigurowane, ale czy mają dane?
        for ev in admin_cfg["key_events"]:
            if events_error:
                fired, status = None, "⚠️ Błąd odczytu danych"
            else:
                fired = event_counts.get(ev, 0)
                status = "❌ Brak danych" if fired == 0 else "✅ OK"
            config_rows.append({
                "MPK": row["MPK"], "Brand": row["Brand"], "Typ": "Key event",
                "Nazwa": ev, "Zdarzenia w okresie": fired, "Status": status,
            })
            if fired == 0:
                dead_params.append({
                    "MPK": row["MPK"], "Brand": row["Brand"],
                    "Typ": "Key event", "Parametr": ev,
                })

        # Custom dimensions: skonfigurowane, ale czy mają wartości w danych?
        for cd in admin_cfg["custom_dimensions"]:
            active_count = core.fetch_custom_dimension_activity(
                pid, cd["parameter_name"], cd["scope"], start, end
            )
            if active_count is None:
                status = "⚠️ Błąd odczytu danych"
            elif active_count == 0:
                status = "❌ Brak danych"
            else:
                status = "✅ OK"
            config_rows.append({
                "MPK": row["MPK"], "Brand": row["Brand"],
                "Typ": f"Custom dimension ({cd['scope']})",
                "Nazwa": cd["parameter_name"], "Zdarzenia w okresie": active_count,
                "Status": status,
            })
            if active_count == 0:
                dead_params.append({
                    "MPK": row["MPK"], "Brand": row["Brand"],
                    "Typ": "Custom dimension", "Parametr": cd["parameter_name"],
                })

    progress.progress(1.0, text="Sprawdzam anomalie w metrykach…")
    anomalies = core.detect_anomalies_for_stores(
        stores, core.MONITORED_METRICS, sigma_threshold=sigma_threshold
    )

    progress.progress(1.0, text="Sprawdzam spójność brandów…")
    brand_issues = []
    stores_by_brand: dict[str, list[str]] = {}
    for _, row in stores.iterrows():
        stores_by_brand.setdefault(row["Brand"], []).append(row["MPK"])

    for brand, mpks in stores_by_brand.items():
        if len(mpks) < 2:
            continue
        for m in mpks:
            others = [x for x in mpks if x != m]
            others_events = [store_events.get(o, set()) for o in others]
            missing = []
            for ev in set().union(*others_events) if others_events else set():
                present_in = sum(1 for oe in others_events if ev in oe)
                # event uznajemy za "oczekiwany" jeśli występuje u ≥połowy pozostałych
                # sklepów brandu, a w tym sklepie go brak
                if present_in / len(others) >= 0.5 and ev not in store_events.get(m, set()):
                    missing.append(ev)
            if missing:
                brand_issues.append({
                    "Brand": brand, "MPK": m,
                    "Liczba brakujących eventów": len(missing),
                    "Brakujące eventy": ", ".join(sorted(missing)),
                })

    progress.empty()

    return {
        "period": {"start": str(start), "end": str(end)},
        "config_rows": config_rows,
        "dead_params": dead_params,
        "anomalies": anomalies,
        "brand_issues": brand_issues,
        "store_events": store_events,
    }


if "audit_result" not in st.session_state:
    st.session_state["audit_result"] = None

if run_clicked:
    stores_to_audit = core.property_map
    if brand_filter:
        stores_to_audit = stores_to_audit[stores_to_audit["Brand"].isin(brand_filter)]
    if stores_to_audit.empty:
        st.error("Brak sklepów dla wybranych filtrów.")
    else:
        st.session_state["audit_result"] = run_full_audit(
            stores_to_audit, lookback_days, sigma_threshold
        )

result = st.session_state["audit_result"]

if result is None:
    st.info("Kliknij **▶️ Uruchom pełny audyt** w panelu bocznym, aby sprawdzić wszystkie sklepy.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# PODSUMOWANIE
# ─────────────────────────────────────────────────────────────
n_dead = len(result["dead_params"])
n_anom = result["anomalies"]["anomalies_found"]
n_brand_issues = len(result["brand_issues"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sklepów zbadanych", len(result["store_events"]))
c2.metric("Martwe parametry/eventy", n_dead)
c3.metric("Anomalie w metrykach", n_anom)
c4.metric("Niespójności brandowe", n_brand_issues)

st.caption(f"Okres sprawdzania aktywności danych: {result['period']['start']} → {result['period']['end']}")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔧 Konfiguracja i martwe parametry",
    "📉 Anomalie w metrykach",
    "🏷️ Spójność brandów",
    "📋 Szczegóły per sklep",
])

with tab1:
    if n_dead:
        st.error(
            f"Znaleziono {n_dead} skonfigurowanych parametrów/eventów bez danych "
            f"w ostatnich {lookback_days} dniach."
        )
    else:
        st.success("Wszystkie skonfigurowane parametry i eventy mają dane.")

    df_cfg = pd.DataFrame(result["config_rows"])
    if not df_cfg.empty:
        only_problems = st.checkbox("Pokaż tylko problemy", value=True)
        view = df_cfg[~df_cfg["Status"].str.startswith("✅")] if only_problems else df_cfg
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Pobierz pełną listę (CSV)",
            df_cfg.to_csv(index=False).encode("utf-8"),
            file_name="ga4_audyt_konfiguracja.csv",
            mime="text/csv",
        )
    else:
        st.info("Brak danych konfiguracyjnych do wyświetlenia.")

with tab2:
    if n_anom:
        st.error(f"Wykryto anomalie w {n_anom} sklepach (próg {sigma_threshold}σ).")
        for a in result["anomalies"]["anomalies"]:
            with st.expander(f"⚠️ {a['MPK']} – {a['Brand']}"):
                for alert in a["alerts"]:
                    label = core.METRIC_LABELS.get(alert["metric"], alert["metric"])
                    st.markdown(
                        f"- **{label}**: {alert['current']} "
                        f"({alert['direction']} średniej {alert['hist_mean']}, "
                        f"{alert['sigma_diff']}σ)"
                    )
    else:
        st.success("Brak anomalii statystycznych w żadnym ze sklepów.")

with tab3:
    if n_brand_issues:
        st.warning(f"{n_brand_issues} przypadków niespójnego wdrożenia eventów wewnątrz brandu.")
        st.dataframe(pd.DataFrame(result["brand_issues"]), use_container_width=True, hide_index=True)
        st.caption(
            "Event uznajemy za 'oczekiwany' dla sklepu, jeśli występuje u co najmniej "
            "połowy pozostałych sklepów tego samego brandu."
        )
    else:
        st.success("Sklepy w ramach każdego brandu mają spójny zestaw eventów.")

with tab4:
    for mpk, events in sorted(result["store_events"].items()):
        row = core.MPK_INDEX.get(mpk)
        brand = row["Brand"] if row is not None else "—"
        with st.expander(f"{mpk} – {brand} ({len(events)} aktywnych eventów)"):
            st.code(", ".join(sorted(events)) if events else "brak danych w okresie", language=None)
