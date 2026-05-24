# Facility Services

Network-tier prototype: ROBOD data over MQTT, Flask backend, SQLite logs, Streamlit dashboard.

1. Start Mosquitto:
   `docker compose up -d`
2. Mac fallback:
   `brew install mosquitto && brew services start mosquitto`
3. Install service dependencies:
   `pip install -r requirements-services.txt`
4. Start backend in terminal 1:
   `python -m services.backend`
5. Start ROBOD publisher in terminal 2:
   `python -m services.bms_simulator`
6. Start dashboard in terminal 3:
   `streamlit run services/dashboard.py`
7. Backend health:
   `curl localhost:8000/api/health`
8. Current state:
   `curl localhost:8000/api/state`
9. Room history:
   `curl "localhost:8000/api/history?room_id=2&hours=24"`

Defaults:
- MQTT: `localhost:1883`
- SQLite: `data/runtime.db`
- Room: `2`
- Simulator speedup: `300x`

Out of scope here: auth, controllers, Kafka, Kubernetes, PostgreSQL, WebSockets.
