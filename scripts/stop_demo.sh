#!/bin/bash
# Shut down the HVAC demo stack started by start_demo.sh

echo "▶ Stopping mosquitto..."
pkill -f "mosquitto -d -c services/mosquitto" 2>/dev/null && echo "  ✓ mosquitto stopped" || echo "  - mosquitto not running"

echo "▶ Stopping Python services..."
pkill -f "services.backend" 2>/dev/null && echo "  ✓ backend stopped" || echo "  - backend not running"
pkill -f "services.bms_simulator" 2>/dev/null && echo "  ✓ simulator stopped" || echo "  - simulator not running"
pkill -f "streamlit run services/dashboard.py" 2>/dev/null && echo "  ✓ dashboard stopped" || echo "  - dashboard not running"

echo ""
echo "✓ All demo processes stopped."
echo "  Close the Terminal tabs manually if needed."
