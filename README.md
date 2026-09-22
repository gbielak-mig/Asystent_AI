# Asystent_AI

Aplikacja Streamlit z agentem AI do Google Analytics 4.

## Strony

- **pages/0_Przeglad.py** – strona startowa, BEZ udziału LLM/Groq (liczona bezpośrednio w kodzie, więc nie zależy od limitów Groq):
  - **Trendy** – który rynek rośnie/spada (wybrana metryka, próg % zmiany tydz./tydz.), wykres per sklep do rozwinięcia,
  - **Nowe analizy** – automatycznie wykryte anomalie (przychód, ruch, konwersje, CR, wsp. odbić, porzucone koszyki) względem 30-dniowej historii.
- **pages/1_Audyt.py** – samodzielny audyt całego portfolio sklepów naraz:
  - konfiguracja GA4 (custom dimensions/metrics, key events z Admin API),
  - czy skonfigurowane parametry/eventy faktycznie zbierają dane („martwe” tagi),
  - anomalie statystyczne w metrykach (odchylenia od 30-dniowej średniej) dla wszystkich sklepów naraz,
  - spójność wdrożenia eventów między sklepami tego samego brandu.
- **Agent_AI.py** – czat z agentem (Groq + tool calling): metryki, trendy, wykresy, ranking sklepów, anomalie na żądanie. Darmowy tier Groq ma niski limit tokenów/minutę (potrafi wywalić się na dłuższych pytaniach) — strona zostaje funkcjonalna, ale priorytetem rozwoju są Przegląd i Audyt, które nie zależą od Groq.

Wspólny kod (klienci GA4/Groq, mapowanie sklepów, pobieranie danych) jest w `ga4_core.py`.

## Uruchomienie

```
pip install -r requirements.txt
streamlit run Agent_AI.py
```

W sidebarze Streamlit strony pojawiają się w kolejności: "0 Przegląd" (startowa), "1 Audyt".

## Wymagania konfiguracyjne

Ten sam service account co dotychczas (scope `analytics.readonly`) wystarcza również do Admin API,
ale w projekcie GCP musi być dodatkowo włączone **Google Analytics Admin API**
(obok już włączonego Google Analytics Data API) — inaczej strona audytu zwróci błąd przy
pobieraniu konfiguracji (widoczny w tabeli jako `⚠️ Błąd Admin API`).

Sekcja "porzucone koszyki" w Przeglądzie wymaga, żeby sklep w ogóle śledził e-commerce
(eventy `add_to_cart`/`purchase`) — dla sklepów bez tego po prostu nie pojawi się tam nic
(bez błędu).

## Klucz LLM

Agent czatu używa **Groq** (darmowy tier) zamiast Anthropic — nie trzeba płatnego klucza API.
W `secrets.toml` sekcja `[anthropic]` została zastąpiona przez:

```toml
[groq]
api_key = "gsk_..."
# model = "openai/gpt-oss-20b"   # opcjonalnie, nadpisuje domyślny model
```

Klucz zakładasz za darmo na [console.groq.com](https://console.groq.com). Pełny format `secrets.toml`
patrz nagłówek `Agent_AI.py`.