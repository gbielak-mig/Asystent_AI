# Asystent_AI

Aplikacja Streamlit z agentem AI do Google Analytics 4.

## Strony

- **Agent_AI.py** – czat z agentem (Claude + tool calling): metryki, trendy, wykresy, ranking sklepów, anomalie na żądanie.
- **pages/1_Audyt.py** – samodzielny audyt całego portfolio sklepów naraz:
  - konfiguracja GA4 (custom dimensions/metrics, key events z Admin API),
  - czy skonfigurowane parametry/eventy faktycznie zbierają dane („martwe” tagi),
  - anomalie statystyczne w metrykach (odchylenia od 30-dniowej średniej) dla wszystkich sklepów naraz,
  - spójność wdrożenia eventów między sklepami tego samego brandu.

Wspólny kod (klienci GA4/Claude, mapowanie sklepów, pobieranie danych) jest w `ga4_core.py`.

## Uruchomienie

```
pip install -r requirements.txt
streamlit run Agent_AI.py
```

Audyt jest widoczny w Streamlit jako osobna strona w sidebarze ("1 Audyt").

## Wymagania konfiguracyjne

Ten sam service account co dotychczas (scope `analytics.readonly`) wystarcza również do Admin API,
ale w projekcie GCP musi być dodatkowo włączone **Google Analytics Admin API**
(obok już włączonego Google Analytics Data API) — inaczej strona audytu zwróci błąd przy
pobieraniu konfiguracji (widoczny w tabeli jako `⚠️ Błąd Admin API`).

`secrets.toml` bez zmian względem dotychczasowego (patrz nagłówek `Agent_AI.py`).