import numpy as np
import math
import math

def normal_cdf_approx(z):
    if z < 0:
        return 1 - normal_cdf_approx(-z)  # symmetry

    t = 1 / (1 + 0.2316419 * z)
    poly = (0.319381530 * t +
            -0.356563782 * t**2 +
            1.781477937 * t**3 +
            -1.821255978 * t**4 +
            1.330274429 * t**5)

    return 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-z**2 / 2) * poly


class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S  #  price
        self.K = K  # strike price
        self.T = T  # Time to expiration in years
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility of asset

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_option_price(self):
        return self.S * normal_cdf_approx(self.d1()) - self.K * np.exp(-self.r * self.T) * normal_cdf_approx(self.d2())

    def put_option_price(self):
        return self.K * np.exp(-self.r * self.T) * normal_cdf_approx(-self.d2()) - self.S * normal_cdf_approx(-self.d1())
