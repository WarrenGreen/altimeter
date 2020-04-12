import json
import random
from math import sin

from altimeter.barometer import Barometer


def sin_gen():
    for x in range(0, 314, 1):
        altitude = sin(x / 100.0) * 100.0
        yield altitude


def linear_gen(altitude_start=1524, altitude_end=2134):
    for altitude in range(altitude_start, altitude_end, 1):
        yield altitude


def main(mode="linear"):
    """

    Args:
        mode: "linear" or "sin"

    """
    output_filepath = f"{mode}_flight.jsonl"
    # All distance variable in meters and pressure in kilopascals
    gps_variance_max = 5
    barometer_variance_max = 0.005
    barometer_start = random.randrange(87 * 10, 108 * 10, 1) / 10.0  # min and max records for sea level pressure
    barometer = Barometer()
    if mode == "linear":
        altitude_gen = linear_gen()
    else:
        altitude_gen = sin_gen()
    with open(output_filepath, "w") as f:
        for altitude in altitude_gen:
            pressure = barometer.pressure(altitude)
            barometer_variance = (
                random.randrange(-barometer_variance_max * 10000, barometer_variance_max * 10000, 1) / 10000.0
            )
            barometer_pressure = pressure + barometer_variance
            barometer_altitude = barometer.altitude(barometer_pressure)

            gps_variance = random.randrange(-gps_variance_max * 10, gps_variance_max * 10, 1) / 10.0 + 0.001
            gps_altitude = altitude + gps_variance
            # Have GPS updates coming less frequently than barometer measurements
            if random.random() < 0.5:
                sample = {
                    "altitude": altitude,
                    "pressure": barometer_pressure,
                    "gps_altitude": gps_altitude,
                    "gps_variance": abs(gps_variance),
                    "barometer_altitude": barometer_altitude,
                }
            else:
                sample = {
                    "altitude": altitude,
                    "pressure": barometer_pressure,
                    "barometer_altitude": barometer_altitude,
                }
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main("linear")
