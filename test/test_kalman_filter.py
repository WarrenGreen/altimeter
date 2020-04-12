import json
import pathlib

from altimeter.barometer import Barometer
from altimeter.kalman_filter import update, predict, gaussian_f

EPSILON = 0.0001
DATA = pathlib.Path(__file__).absolute().parent / "data"


def test_update():
    assert update(14, 2, 95, 4) == (41.0, 1.3333333333333333)
    assert update(8, 3, 99, 3) == (53.5, 1.5)
    assert update(23, 6, 65, 2) == (54.5, 1.5)
    assert update(77, 6, 101, 4) == (91.4, 2.4000000000000004)
    assert update(57, 9, 119, 2) == (107.72727272727273, 1.6363636363636362)
    assert update(46, 1, 108, 3) == (61.5, 0.75)
    assert update(20, 9, 120, 5) == (84.28571428571429, 3.2142857142857144)
    assert update(69, 5, 92, 3) == (83.375, 1.875)
    assert update(56, 2, 111, 3) == (78.0, 1.2000000000000002)
    assert update(95, 9, 59, 4) == (70.07692307692308, 2.769230769230769)


def test_predict():
    assert predict(63, 8, 119, 2) == (182, 10)
    assert predict(77, 1, 106, 3) == (183, 4)
    assert predict(91, 10, 80, 5) == (171, 15)
    assert predict(51, 1, 52, 4) == (103, 5)
    assert predict(9, 6, 59, 5) == (68, 11)
    assert predict(38, 3, 113, 5) == (151, 8)
    assert predict(51, 2, 87, 2) == (138, 4)
    assert predict(38, 8, 96, 3) == (134, 11)
    assert predict(71, 4, 53, 5) == (124, 9)
    assert predict(9, 10, 69, 2) == (78, 12)


def test_gaussian_f():
    mu = 63
    var = 8
    auc = 0.0
    for x in range(mu - (var * 2), mu + (var * 2), 1):
        auc += gaussian_f(mu, var, x)

    assert auc - 1.0 < EPSILON


def test_kalman_filter_linear_data():
    barometer = Barometer()
    mu = 0
    sigma = 1000
    prev_mu = mu
    prev_sigma = sigma
    with open(str(DATA / "linear_flight.jsonl"), "r") as f:
        for line in f:
            sample = json.loads(line)
            gps_altitude = sample["gps_altitude"]
            gps_sigma = sample["gps_variance"]
            mu, sigma = update(mu, sigma, gps_altitude, gps_sigma)
            barometer_altitude = barometer.altitude(sample["pressure"])
            mu, sigma = update(mu, sigma, barometer_altitude, barometer.sigma)

            print(f"Update: [{mu}, {sigma}], true altitude {sample['altitude']}")
            delta = mu - prev_mu
            pred_mu, pred_sigma = predict(mu, sigma, delta, prev_sigma)
            prev_mu = mu
            prev_sigma = sigma
            mu = pred_mu
            sigma = pred_sigma

        final_altitude = sample["altitude"]
        # print the final, resultant mu, sig
        print("\n")
        print(f"Final result: [{mu}, {sigma}], true altitude {final_altitude}")

        # assert that final prediction is within 5% of true altitude
        assert mu - final_altitude < final_altitude * 0.05


if __name__ == "__main__":
    test_kalman_filter_linear_data()
