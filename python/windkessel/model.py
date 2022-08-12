import torch
import numpy
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from .utils import *


class WindkesselBaseModel():
    """
    Модель windkessel.
    """

    def __init__(self):
        self.R = 0.5      # mmHg * s / mL
        self.Z_0 = 0.0485 # mmHg * s / mL
        self.C = 2.27     # mL / mmHg

        self.P_out = 33.2 # mmHg

        # Отсчёты по времени для графика.
        self.T = None

        # Графики.
        self.P = None
        self.der_P = None

        self.Q_in = None
        self.der_Q_in = None


    def set_P(self, new_T, new_P, new_der_P=None):
        """
        Задание графика P.
        """

        self.T = new_T
        self.P = new_P
        if new_der_P is None:
            self.der_P = calc_der(self.T, self.P)
        else:
            self.der_P = new_der_P

        self.get_P = loop_function(interp1d(self.T, self.P), min(self.T), max(self.T))
        self.get_der_P = loop_function(interp1d(self.T, self.der_P), min(self.T), max(self.T))


    def set_Q_in(self, new_T, new_Q_in, new_der_Q_in):
        """
        Задание графика Q_in.
        """

        self.T = new_T
        self.Q_in = new_Q_in
        if new_der_P is None:
            self.der_Q_in = calc_der(self.T, self.Q_in)
        else:
            self.der_Q_in = new_der_Q_in

        self.get_Q_in = loop_function(interp1d(self.T, self.Q_in), min(self.T), max(self.T))
        self.get_der_Q_in = loop_function(interp1d(self.T, self.der_Q_in), min(self.T), max(self.T))


    def P_rhs_(self, P, Q_in, der_Q_in):
        """
        Формула правой части для задачи dP / dt = ...
        """

        return (self.P_out - P + (self.Z_0 + self.R) * Q_in) / (self.R * self.C) + self.Z_0 * der_Q_in


    def Q_in_rhs_(self, P, der_P, Q_in):
        """
        Формула правой части для задачи dQ_in / dt = ...
        """

        return (der_P + (P - self.P_out - (self.Z_0 + self.R) * Q_in) / (self.R * self.C)) / self.Z_0


    def P_rhs(self, t, P):
        """
        Правая часть для задачи dP / dt = ...
        """

        raise NotImplementedError


    def Q_in_rhs(self, t, Q_in):
        """
        Правая часть для задачи dQ_in / dt = ...
        """

        raise NotImplementedError


    ########################
    ## ПОДБОР ПАРАМЕТРОВ. ##
    ########################

    def get_diastole_start(self):
        """
        Возвращает индекс, соответствующий времени максимального давления, а также само время.
        """

        # Поиск взвешенного минимума производной dP / dt.
        lvet_array = self.der_P * (0.5 - numpy.abs(0.5 - self.T / self.T[-1]))**2
        min_index = numpy.argmin(lvet_array)

        return min_index, self.T[min_index]


    @staticmethod
    def diastole_exp_decay(t, P_0, RC, P_out):
        return P_0 * numpy.exp(-t / RC) + P_out


    def get_exp_param(self):
        """
        Возвращает P_0, R*C, P_out
        """

        # Время начала диастолы.
        ds_index, ds_time = self.get_diastole_start()
        #ds_index += (self.T.shape[0] - ds_index) // 5

        # Оценки параметров.
        RC = self.R * self.C
        P_0 = (self.P[ds_index] - self.P_out) * numpy.exp(self.T[ds_index] / RC)
        #P_0 = self.P_out
        #p_lvet = self.P[ds_index]

        # Подгон кривой по МНК.
        fit_param, fit_covariance = curve_fit(self.diastole_exp_decay, self.T[ds_index:], self.P[ds_index:], p0=[P_0, RC, self.P_out])
        #fit_param, fit_covariance = curve_fit(lambda x, a, c: parabola(x, a, b_fixed, c), x, y)
        #fit_param, _ = curve_fit(lambda t, RC, P_out: self.diastole_exp_decay_without_P0(t, p_lvet, RC, P_out), self.T[t0:], self.P[t0:], p0 = [RC, self.P_out])

        # Проверка корректности.
        if fit_param[2] <= 0.0:
            diastole_exp_decay_without_P_out = lambda t, P_0, RC : self.diastole_exp_decay(t, P_0, RC, self.P_out)
            fit_param, fit_covariance = curve_fit(diastole_exp_decay_without_P_out, self.T[ds_index:], self.P[ds_index:], p0=[P_0, RC])
            fit_param = numpy.append(fit_param, self.P_out)

        return fit_param



class WindkesselModel(WindkesselBaseModel):
    """
    Модель windkessel, написанная на NumPy.
    """

    def __init__(self):
        WindkesselBaseModel.__init__(self)


    def P_rhs(self, t, P):
        """
        Правая часть для задачи dP / dt = ...
        """

        Q_in = self.get_Q_in(t)
        der_Q_in = self.get_der_Q_in(t)

        return self.P_rhs_(P, Q_in, der_Q_in)


    def Q_in_rhs(self, t, Q_in):
        """
        Правая часть для задачи dQ_in / dt = ...
        """

        P = self.get_P(t)
        der_P = self.get_der_P(t)

        return self.Q_in_rhs_(P, der_P, Q_in)


    def forward():
        pass



class WindkesselTorchModel(WindkesselBaseModel, torch.nn.Module):
    """
    Модель windkessel, написанная на PyTorch.
    """

    def __init__(self):
        WindkesselBaseModel.__init__(self)
        torch.nn.Module.__init__(self)

        self.R   = torch.nn.Parameter(torch.ones(1, dtype=torch.float64) * self.R)
        self.Z_0 = torch.nn.Parameter(torch.ones(1, dtype=torch.float64) * self.Z_0)
        self.C   = torch.nn.Parameter(torch.ones(1, dtype=torch.float64) * self.C)

        self.P_out = torch.nn.Parameter(torch.ones(1, dtype=torch.float64) * self.P_out)


    def P_rhs(self, t, P):
        """
        Правая часть для задачи dP / dt = ...
        """

        P = torch.from_numpy(P)

        Q_in = torch.ones(1, dtype=torch.float64) * self.get_Q_in(t)
        der_Q_in = torch.ones(1, dtype=torch.float64) * self.get_der_Q_in(t)

        return self.P_rhs_(P, Q_in, der_Q_in)


    def Q_in_rhs(self, t, Q_in):
        """
        Правая часть для задачи dQ_in / dt = ...
        """

        Q_in = torch.from_numpy(Q_in)

        P = torch.ones(1, dtype=torch.float64) * self.get_P(t)
        der_P = torch.ones(1, dtype=torch.float64) * self.get_der_P(t)

        return self.Q_in_rhs_(P, der_P, Q_in)


    def forward():
        pass
