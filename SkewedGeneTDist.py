import numpy as np
from math import comb, sqrt
from scipy.special import beta

class SkewedGeneTDist:
    def __init__(self, mu, sig, l, p, q) -> None:
        assert sig>0
        assert 1>l>-1
        assert p>0
        assert q>0
        self.mu = mu
        self.sig = sig
        self.l = l
        self.p = p
        self.q = q
        self.v = self._make_v()
        self.m = self._make_m()

    def _make_v(self):
        v_num = self.q ** (-1 / self.p)
        v_den1 = (1 + 3 * self.l ** 2) * beta(3 / self.p, self.q - 2 / self.p) / beta(1 / self.p, self.q)
        v_den2 = 4 * self.l ** 2 * (beta(2 / self.p, self.q - 1 / self.p) ** 2) / (beta(1 / self.p, self.q) ** 2)
        v = v_num / sqrt(v_den1 - v_den2)
        return v

    def _make_m(self):
        m_mul = self.l * self.v * self.sig
        m_num = 2 * self.q ** (1 / self.p) * beta(2 / self.p, self.q - 1 / self.p)
        m_den = beta(1 / self.p, self.q)
        m = m_mul * m_num / m_den
        return m

    def moment(self, h):
        moment = 0
        for r in range(h+1):
            mul_mul = (1 + self.l) ** (r + 1) + (-1) ** r * (1 - self.l) ** (r + 1)
            mul = comb(h, r) * mul_mul * (-self.l) ** (h - r)

            num_mul = (self.v * self.sig) ** h * self.q ** (h / self.p)
            num = num_mul * beta((r + 1) / self.p, self.q - r / self.p) * beta(2 / self.p, self.q - 1 / self.p) ** (
                        h - r)
            den = 2 ** (r - h + 1) * beta(1 / self.p, self.q) ** (h - r + 1)
            to_add = mul * num / den
            # print(comb(h, r), mul_mul, (-self.l) ** (h - r))
            moment = moment + to_add
        return moment

    def pdf(self, x):
        den1 = 2 * self.v * self.sig * self.q ** (1 / self.p) * beta(1 / self.p, self.q)
        den2_num = abs(x - self.mu + self.m) ** self.p
        den2_den = self.q * (self.v * self.sig) ** self.p * (1 + self.l * np.sign(x - self.mu + self.m)) ** self.p
        den = den1 * (1 + den2_num / den2_den) ** (1 / self.p + self.q)
        return self.p / den



