DEFAULT_SEA_LEVEL_PRESSURE = 101.325  # kilopascals
DEFAULT_SEA_LEVEL_TEMP = 15  # celsius


class Barometer:
    def __init__(
        self,
        sea_level_pressure=DEFAULT_SEA_LEVEL_PRESSURE,
        sea_level_temp=DEFAULT_SEA_LEVEL_TEMP,
        variance=0.5,
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
        return self.sea_level_pressure * pow((1 - 2.25577e-5 * altitude), 5.25588)

    def altitude(self, pressure):
        return (1 - pow(pressure / self.sea_level_pressure, 1 / 5.25588)) / 2.25577e-5
