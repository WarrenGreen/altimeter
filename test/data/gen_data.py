import json
import random


def main():
    output_filepath = "linear_flight.jsonl"
    # All distance variable in meters and pressure in kilopascals
    gps_variance_max = 5
    barometer_variance_max = 0.5
    barometer_start = random.randrange(87 * 10, 108 * 10, 1) / 10.0  # min and max records for sea level pressure
    altitude_start = 1524
    sea_level_pressure = 101.325
    with open(output_filepath, "w") as f:
        for altitude in range(altitude_start, 2134, 1):
            pressure = sea_level_pressure * pow((1 - 2.25577e-5 * altitude), 5.25588)
            barometer_variance = (
                random.randrange(-barometer_variance_max * 100, barometer_variance_max * 100, 1) / 100.0
            )
            barometer_pressure = pressure + barometer_variance
            gps_variance = random.randrange(-gps_variance_max * 10, gps_variance_max * 10, 1) / 10.0 + 0.001
            gps_altitude = altitude + gps_variance
            sample = {"altitude": altitude, "pressure": barometer_pressure, "gps_altitude": gps_altitude, "gps_variance": abs(gps_variance)}
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
