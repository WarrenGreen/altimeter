import json
import random
from math import sin, cos

import numpy as np

from altimeter.barometer import Barometer
from altimeter.smoothing import smooth


def sin_gen():
    for x in range(0, 314, 1):
        altitude = sin(x / 100.0) * 100.0
        yield altitude


def linear_gen(altitude_start=1524, altitude_end=2134):
    for altitude in range(altitude_start, altitude_end, 1):
        yield altitude


def generate_polynomial(max_poly_size=4):
    poly_size = random.randint(1, max_poly_size)

    def poly(x):
        poly_value = 0
        for poly_index in range(2, poly_size + 1):
            if poly_index % 2 == 0:
                poly_value += pow(sin(x), poly_index)
            else:
                poly_value += pow(cos(x), poly_index)

        slope_dir = 1 if random.random() > 0.5 else -1
        poly_value *= slope_dir
        return poly_value

    return poly


def poly_gen(n=500):
    poly = generate_polynomial()
    altitude = 0.0
    altitudes = []
    for index in range(n):
        if index % 6:
            poly = generate_polynomial()
        time = index / 100.0
        altitude_delta = poly(time)
        altitude += altitude_delta
        altitudes.append(altitude)

    smoothed_altitudes = smooth(np.array(altitudes))
    for altitude in smoothed_altitudes:
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
    elif mode == "sine":
        altitude_gen = sin_gen()
    else:
        altitude_gen = poly_gen()
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
    main("poly")
