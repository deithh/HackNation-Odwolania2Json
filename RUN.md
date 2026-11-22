## Uruchomienie

cd /opt/solution && source .venv/bin/activate && python main.py --input /opt/solution/input --output /opt/solution/output --workers 2

---

#### Zewnętrzne API i polityka prywatności

- Dostawca: **Regolo.ai** (Seeweb S.r.l., Włochy / UE)
- Endpointy:
  - `https://api.regolo.ai/v1/chat/completions` (model: `qwen3.5-9b`)
  - `https://api.regolo.ai/v1/systemone` (model: `clef-flash`)
- Klucze API: `/opt/solution/.env`
- Polityka prywatności: https://regolo.ai/privacy-policy/
