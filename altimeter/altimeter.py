from altimeter.barometer import Barometer
from altimeter.kalman_filter import KalmanFilter
from altimeter.model.linear_model import LinearModel


class Altimeter:
    def __init__(self, motion_model_class=LinearModel):
        self.barometer = Barometer()
        self.motion_model = motion_model_class()
        self.prev_altitudes = [0]
        self.prev_variances = [1000]

    def add_barometer_measurement(self, pressure, time=None):
        barometer_altitude = self.barometer.altitude(pressure)
        if time is None or time >= len(self.prev_altitudes):
            mu, sigma = KalmanFilter.update(self.mu, self.sigma, barometer_altitude, self.barometer.variance)
            self.prev_altitudes.append(mu)
            self.prev_variances.append(sigma)
            self.predict()
        else:
            self.prev_altitudes[time], self.prev_variances[time] = KalmanFilter.update(self.prev_altitudes[time], self.prev_variances[time], barometer_altitude, self.barometer.variance)

    def add_gps_measurement(self, altitude, variance, time=None):
        if time is None or time >= len(self.prev_altitudes):
            mu, sigma = KalmanFilter.update(self.mu, self.sigma, altitude, variance)
            self.prev_altitudes.append(mu)
            self.prev_variances.append(sigma)
            self.predict()
        else:
            self.prev_altitudes[time], self.prev_variances[time] = KalmanFilter.update(self.prev_altitudes[time],
                                                                                       self.prev_variances[time],
                                                                                       altitude,
                                                                                       variance)

    def predict(self):
        _, delta = self.motion_model.predict(self.prev_altitudes)
        mu, sigma = KalmanFilter.predict(self.mu, self.sigma, delta, self.motion_model.variance)
        self.prev_altitudes.append(mu)
        self.prev_variances.append(sigma)

    @property
    def mu(self):
        return self.prev_altitudes[-1]

    @property
    def sigma(self):
        return self.prev_variances[-1]
