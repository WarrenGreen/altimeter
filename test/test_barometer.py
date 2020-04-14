import json
import pathlib

from altimeter.barometer import Barometer

EPSILON = 0.0001
DATA = pathlib.Path(__file__).absolute().parent / "data"


def test_default_barometer():
    barometer = Barometer()
    with open(str(DATA / "linear_flight.jsonl"), "r") as f:
        for line in f:
            sample = json.loads(line)
            true_altitude = sample["altitude"]
            pressure = sample["pressure"]
            barometer_altitude = barometer.altitude(pressure)
            assert abs(barometer_altitude - true_altitude) < 1
            assert abs(pressure - barometer.pressure(barometer_altitude)) < EPSILON
