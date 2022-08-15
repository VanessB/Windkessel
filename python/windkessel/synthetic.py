import numpy


class SyntheticArtery():
    def __init__(self):
        self.T_max = 0.09 # Время достижения максимума, с.
        self.T_s   = 0.33 # Время конца систолы, с.
        self.T_d   = 0.35 # Время начала диастолы, с.
        self.T     = 0.9  # Продолжительность цикла, с.
        self.Q_max = 370  # Максимальный поток, мл/с.
        self.R_f   = 0.3  # КОэффициент отражения.

    def get_Q_LA(self, t):
        t = t % self.T

        if 0.0 <= t and t < self.T_max:
            return 0.5 * self.Q_max * (numpy.sin(numpy.pi * (t - self.T_max / 2) / self.T_max) + 1)
        elif t < self.T_s:
            return 0.5 * self.Q_max * (numpy.cos(numpy.pi * (t - self.T_max) / (self.T_s - self.T_max)) + 1)
        else:
            return 0.0


    def get_Q_in(self, t):
        return self.get_Q_LA(t) + self.R_f * self.get_Q_LA(t - self.T_d)
