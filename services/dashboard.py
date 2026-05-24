"""Streamlit dashboard for live facility-team HVAC state."""

from __future__ import annotations

import time

import altair as alt
import pandas as pd
import requests
import streamlit as st

from src.config import CONFIG


BACKEND_URL = "http://localhost:8000"


def main() -> None:
    st.set_page_config(page_title="Smart Building HVAC", layout="wide")
    st.title("Smart Building HVAC Network Tier")

    placeholder = st.empty()
    while True:
        state = _get_json("/api/state", {"rooms": {}})
        health = _get_json("/api/health", {})

        with placeholder.container():
            _render_overview(state)
            _render_detail(state)
            _render_health(health, state)

        time.sleep(2)


def _render_overview(state: dict) -> None:
    st.header("Building Overview")
    rooms = state.get("rooms", {})
    if not rooms:
        st.info("Waiting for MQTT sensor data...")
        return

    cols = st.columns(max(1, len(rooms)))
    for col, (room_id, room_state) in zip(cols, sorted(rooms.items(), key=lambda item: int(item[0]))):
        indoor = room_state.get("indoor_c")
        comfort_ok = (
            indoor is not None
            and CONFIG["comfort_min_c"] < float(indoor) < CONFIG["comfort_max_c"]
        )
        color = "#0f8a4b" if comfort_ok else "#b42318"
        with col:
            st.markdown(
                f"""
                <div style="border-left: 6px solid {color}; padding: 0.5rem 0.75rem; background: #f7f7f7;">
                  <h3 style="margin: 0;">Room {room_id}</h3>
                  <p>Occupancy: <b>{room_state.get("occupancy", "-")}</b></p>
                  <p>Indoor: <b>{_fmt(indoor)} °C</b></p>
                  <p>Predicted in 2h: <b>{_fmt(room_state.get("predicted_count"))}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_detail(state: dict) -> None:
    st.header("Per-Room Detail")
    rooms = sorted(state.get("rooms", {}).keys(), key=int)
    if not rooms:
        return

    selected = st.selectbox("Room", rooms, index=0)
    history = _get_json(f"/api/history?room_id={selected}&hours=24", {"history": []})
    df = pd.DataFrame(history.get("history", []))
    if df.empty:
        st.info("No history yet for this room.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    count_df = df.melt(
        id_vars=["timestamp"],
        value_vars=["occupancy_count", "predicted_count"],
        var_name="series",
        value_name="count",
    )
    count_chart = (
        alt.Chart(count_df)
        .mark_line()
        .encode(x="timestamp:T", y="count:Q", color="series:N")
        .properties(height=240)
    )
    st.altair_chart(count_chart, use_container_width=True)

    temp_df = df.melt(
        id_vars=["timestamp"],
        value_vars=["indoor_c", "outdoor_c"],
        var_name="series",
        value_name="temperature_c",
    )
    temp_chart = (
        alt.Chart(temp_df)
        .mark_line()
        .encode(x="timestamp:T", y="temperature_c:Q", color="series:N")
        .properties(height=240)
    )
    st.altair_chart(temp_chart, use_container_width=True)


def _render_health(health: dict, state: dict) -> None:
    st.header("System Health")
    rooms = state.get("rooms", {})
    last_updates = [
        room_state.get("last_update")
        for room_state in rooms.values()
        if room_state.get("last_update")
    ]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MQTT connected", str(health.get("mqtt_connected", False)))
    c2.metric("Rooms seen", len(health.get("rooms_seen", [])))
    c3.metric("Sensor records", health.get("sensor_records", 0))
    c4.metric("Prediction records", health.get("prediction_records", 0))
    st.caption(f"Last update: {max(last_updates) if last_updates else '-'}")


def _get_json(path: str, default: dict) -> dict:
    try:
        response = requests.get(f"{BACKEND_URL}{path}", timeout=1.0)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return default


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}"


if __name__ == "__main__":
    main()
