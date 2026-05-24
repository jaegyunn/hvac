CREATE TABLE IF NOT EXISTS sensor_log (
    timestamp TEXT,
    room_id INT,
    occupancy_count INT,
    indoor_c REAL,
    outdoor_c REAL,
    PRIMARY KEY (timestamp, room_id)
);

CREATE TABLE IF NOT EXISTS prediction_log (
    timestamp TEXT,
    room_id INT,
    horizon_minutes INT,
    predicted_count REAL,
    PRIMARY KEY (timestamp, room_id)
);
