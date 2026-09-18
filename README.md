# Asystent_AI

Aplikacja Streamlit z agentem AI do Google Analytics 4.

## Strony

- **Agent_AI.py** – czat z agentem (Groq + tool calling): metryki, trendy, wykresy, ranking sklepów, anomalie na żądanie.
- **pages/1_Audyt.py** – samodzielny audyt całego portfolio sklepów naraz:
  - konfiguracja GA4 (custom dimensions/metrics, key events z Admin API),
  - czy skonfigurowane parametry/eventy faktycznie zbierają dane („martwe” tagi),
  - anomalie statystyczne w metrykach (odchylenia od 30-dniowej średniej) dla wszystkich sklepów naraz,
  - spójność wdrożenia eventów między sklepami tego samego brandu.

Wspólny kod (klienci GA4/Groq, mapowanie sklepów, pobieranie danych) jest w `ga4_core.py`.

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

## Klucz LLM

Agent czatu używa **Groq** (darmowy tier) zamiast Anthropic — nie trzeba płatnego klucza API.
W `secrets.toml` sekcja `[anthropic]` została zastąpiona przez:

```toml
[groq]
api_key = "gsk_..."
# model = "llama-3.3-70b-versatile"   # opcjonalnie, nadpisuje domyślny model
```

Klucz zakładasz za darmo na [console.groq.com](https://console.groq.com). Pełny format `secrets.toml`
patrz nagłówek `Agent_AI.py`.