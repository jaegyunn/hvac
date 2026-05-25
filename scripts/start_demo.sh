#!/bin/bash
# Boot the full HVAC demo stack: mosquitto + backend + bms_simulator + dashboard
# Opens 3 Terminal tabs for the Python services and runs mosquitto as a background daemon.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "▶ Cleaning up any previous demo processes..."
pkill -f "mosquitto -d -c services/mosquitto" 2>/dev/null || true
pkill -f "services.backend" 2>/dev/null || true
pkill -f "services.bms_simulator" 2>/dev/null || true
pkill -f "streamlit run services/dashboard.py" 2>/dev/null || true
sleep 1

echo "▶ Starting Mosquitto MQTT broker (background)..."
mosquitto -d -c services/mosquitto/config/mosquitto.conf
echo "  ✓ Mosquitto running on port 1883"

echo ""
echo "▶ Opening 3 Terminal tabs (backend → simulator → dashboard)..."

osascript <<EOF
tell application "Terminal"
    activate
    do script "cd '$PROJECT_DIR' && echo '== Backend ==' && python -m services.backend"
    delay 0.5
    do script "cd '$PROJECT_DIR' && sleep 3 && echo '== BMS Simulator (rooms 1-5, test split) ==' && python -m services.bms_simulator --rooms 1,2,3,4,5 --start-from-test"
    delay 0.5
    do script "cd '$PROJECT_DIR' && sleep 5 && echo '== Streamlit Dashboard ==' && streamlit run services/dashboard.py"
end tell
EOF

echo ""
echo "✓ Demo stack starting."
echo "  Dashboard URL: http://localhost:8501 (opens automatically in ~10s)"
echo ""
echo "To stop everything: bash scripts/stop_demo.sh"
