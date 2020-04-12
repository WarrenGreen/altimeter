DEFAULT_SEA_LEVEL_PRESSURE = 101.325  # kilopascals
DEFAULT_SEA_LEVEL_TEMP = 15.0  # celsius
KELVIN_CONSTANT = 273.15


class Barometer:
    def __init__(
        self, sea_level_pressure=DEFAULT_SEA_LEVEL_PRESSURE, sea_level_temp=DEFAULT_SEA_LEVEL_TEMP, variance=0.5
    ):
        """

        Args:
            sea_level_pressure (float): pressure at sea level in kPa
            sea_level_temp (float): temperature at sea level in celsius
            variance (float): variance in kPa
        """
        self.sea_level_pressure = sea_level_pressure
        self.sea_level_temp = sea_level_temp
        self.variance = variance

    def pressure(self, altitude):
        return self.sea_level_pressure * pow(
            1 - ((0.0065 * altitude) / (self.sea_level_temp + (0.0065 * altitude) + KELVIN_CONSTANT)), 5.25588
        )

    def altitude(self, pressure):
        return (
            (pow(self.sea_level_pressure / pressure, 1 / 5.25588) - 1) * (self.sea_level_temp + KELVIN_CONSTANT)
        ) / 0.0065
