import torch
import numpy
from scipy.interpolate import interp1d


def calc_der(T, X):
    """
    Вычисление первой производной по времени.
    """

    der_X = numpy.zeros_like(X)

    # Производные на границах.
    der_X[0] = (X[1] - X[0]) / (T[1] - T[0])
    der_X[-1] = (X[-1] - X[-2]) / (T[-1] - T[-2])

    # Производные внутри.
    for index in range(1, X.shape[0] - 1):
        der_X[index] = (X[index + 1] - X[index - 1]) / (T[index + 1] - T[index - 1])

    return der_X



def loop_function(func, x_min, x_max):
    """
    Зацикливание одномерной функции.
    """

    return lambda x : func((x - x_min) % (x_max - x_min) + x_min)



class WindkesselModel(torch.nn.Module):
    """
    Модель windkessel.
    """

    def __init__(self):
        super().__init__()

        self.R   = torch.nn.Parameter(torch.ones(1, dtype=torch.float64) * 0.5)    # mmHg * s / mL
        self.Z_0 = torch.nn.Parameter(torch.ones(1, dtype=torch.float64) * 0.0485) # mmHg * s / mL
        self.C   = torch.nn.Parameter(torch.ones(1, dtype=torch.float64) * 2.27)   # mL / mmHg

        self.P_out = torch.nn.Parameter(torch.ones(1, dtype=torch.float64) * 33.2) # mmHg

        # Отсчёты по времени для графика.
        self.T = None

        # Графики.
        self.P_ = None
        self.der_P_ = None

        self.Q_in_ = None
        self.der_Q_in_ = None


    def set_P(self, new_T, new_P):
        """
        Задание графика P.
        """

        self.T = new_T
        self.P = new_P
        self.der_P = calc_der(self.T, self.P)

        self.get_P = loop_function(interp1d(self.T, self.P), numpy.min(self.T), numpy.max(self.T))
        self.get_der_P = loop_function(interp1d(self.T, self.der_P), numpy.min(self.T), numpy.max(self.T))


    def set_Q_in(self, new_T, new_Q_in):
        """
        Задание графика Q_in.
        """

        self.T = new_T
        self.Q_in = new_Q_in
        self.der_Q_in = calc_der(self.T, self.Q_in)

        self.get_Q_in = loop_function(interp1d(self.T, self.Q_in), numpy.min(self.T), numpy.max(self.T))
        self.get_der_Q_in = loop_function(interp1d(self.T, self.der_Q_in), numpy.min(self.T), numpy.max(self.T))


    def P_rhs(self, t, P):
        """
        Правая часть для задачи dP / dt = ...
        """

        P = torch.from_numpy(P)

        Q_in = torch.ones(1, dtype=torch.float64) * self.get_Q_in(t)
        der_Q_in = torch.ones(1, dtype=torch.float64) * self.get_der_Q_in(t)

        result = (self.P_out - P + (self.Z_0 + self.R) * Q_in) / (self.R * self.C) + self.Z_0 * der_Q_in
        return result


    def Q_in_rhs(self, t, Q_in):
        """
        Правая часть для задачи dQ_in / dt = ...
        """

        Q_in = torch.from_numpy(Q_in)

        P = torch.ones(1, dtype=torch.float64) * self.get_P(t)
        der_P = torch.ones(1, dtype=torch.float64) * self.get_der_P(t)

        result = (der_P + (P - self.P_out - (self.Z_0 + self.R) * Q_in) / (self.R * self.C)) / self.Z_0
        return result


    def forward():
        pass
