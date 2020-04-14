import json
import pathlib

import numpy as np

from altimeter.altimeter import Altimeter
from altimeter.smoothing import smooth

EPSILON = 0.0001
DATA = pathlib.Path(__file__).absolute().parent / "data"


def test_altimeter_linear_data_linear_model(plot=False):
    altimeter = Altimeter()

    # For plotting
    true_altitudes = []
    gps_altitudes = []
    barometer_altitudes = []
    with open(str(DATA / "linear_flight.jsonl"), "r") as f:
        for time, line in enumerate(f):
            sample = json.loads(line)
            if "gps_altitude" in sample:
                gps_altitude = sample["gps_altitude"]
                gps_sigma = sample["gps_variance"]
                altimeter.add_gps_measurement(gps_altitude, gps_sigma, time)
            barometer_pressure = sample["pressure"]
            barometer_altitude = altimeter.barometer.altitude(barometer_pressure)
            altimeter.add_barometer_measurement(barometer_pressure, time)

            if plot:
                true_altitudes.append(sample["altitude"])
                if "gps_altitude" in sample:
                    gps_altitudes.append(gps_altitude)
                else:
                    gps_altitudes.append(None)
                barometer_altitudes.append(barometer_altitude)

        final_altitude = sample["altitude"]
        # print the final, resultant mu, sig
        print("\n")
        print(f"Final result: [{altimeter.mu}, {altimeter.sigma}], true altitude {final_altitude}")

        if plot:
            import matplotlib.pyplot as plt
            pred_altitudes = altimeter.prev_altitudes[1:]
            smooth_pred_altitudes = smooth(np.array(pred_altitudes), window_len=3, window="flat")
            plt.figure()
            plt.plot(gps_altitudes, "k+", label="gps measurements")
            plt.plot(barometer_altitudes, "r+", label="barometer measurements")
            plt.plot(pred_altitudes, "b-", label="predicted altitudes")
            plt.plot(smooth_pred_altitudes, "m-", label="smoothed predicted altitudes")
            plt.plot(true_altitudes, "g-", label="true altitudes")
            plt.title("Estimated Altitude", fontweight="bold")
            plt.xlabel("Time Step")
            plt.ylabel("Altitude (m)")
            plt.legend()
            plt.show()

        # assert that final prediction is within 5% of true altitude
        assert altimeter.mu - final_altitude < final_altitude * 0.05


if __name__ == "__main__":
    test_altimeter_linear_data_linear_model(plot=True)
