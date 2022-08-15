import torch
import numpy

from scipy.integrate import simps, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize

from .utils import *
from .synthetic import SyntheticArtery


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


    def set_Q_in(self, new_T, new_Q_in, new_der_Q_in=None):
        """
        Задание графика Q_in.
        """

        self.T = new_T
        self.Q_in = new_Q_in
        if new_der_Q_in is None:
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
        Возвращает индекс, соответствующий времени начала диастолы, а также само время.
        """

        # Поиск взвешенного минимума производной dP / dt.
        lvet_array = self.der_P * (0.5 - numpy.abs(0.5 - self.T / self.T[-1]))**2
        ds_index = numpy.argmin(lvet_array)

        return ds_index, self.T[ds_index]


    def get_exp_decay_start(self):
        """
        Возвращает индекс, соответствующий времени начала экспоненциального убывания, а также само время.
        """

        ds_index, ds_time = self.get_diastole_start()

        eds_time = (2.0 * ds_time + self.T[-1]) / 3.0
        eds_index = numpy.searchsorted(self.T, eds_time)
        eds_time = self.T[eds_index]

        return eds_index, eds_time


    @staticmethod
    def diastole_exp_decay(t, P_0, RC, P_out):
        return P_0 * numpy.exp(-t / RC) + P_out


    def get_exp_param(self):
        """
        Возвращает P_0, R*C, P_out
        """

        # Время начала экспоненциального спада.
        eds_index, eds_time = self.get_exp_decay_start()

        # Оценки параметров.
        RC = self.R * self.C
        P_0 = (self.P[eds_index] - self.P_out) * numpy.exp(self.T[eds_index] / RC)
        #P_0 = self.P_out
        #p_lvet = self.P[ds_index]

        # Подгон кривой по МНК.
        fit_param, fit_covariance = curve_fit(self.diastole_exp_decay, self.T[eds_index:], self.P[eds_index:], p0=[P_0, RC, self.P_out])

        # Проверка корректности.
        if fit_param[2] <= 0.0:
            # Фиксированное значение P_out
            fixed_P_out = 0.0
            #fixed_P_out = self.P_out

            # Подгоняемая функция с фиксированным значением.
            diastole_exp_decay_without_P_out = lambda t, P_0, RC : self.diastole_exp_decay(t, P_0, RC, fixed_P_out)

            fit_param, fit_covariance = curve_fit(diastole_exp_decay_without_P_out, self.T[eds_index:], self.P[eds_index:], p0=[P_0, RC])
            fit_param = numpy.append(fit_param, fixed_P_out)
            fit_covariance = numpy.append(fit_param, 0.0)

        return fit_param, fit_covariance


    def get_C_from_SV(self, SV):
        """
        Получение C по сердечному выбросу.
        """

        pulse_pressure = max(self.P) - min(self.P)
        return pulse_pressure / SV


    def get_R_from_SV(self, SV):
        """
        Получение C по сердечному выбросу.
        """

        mean_Q_in = SV / (self.T[-1] - self.T[0])
        mean_P = simps(self.P, self.T)

        return (mean_P - self.P_out) / mean_Q_in


    @staticmethod
    def P_functional(x, T, P, solve_ivp_params={}):
        """
        Функционал невязки между заданным и синтетическим P.
        """

        # Сдвиг временной сетки в нулевую точку.
        T = T - T[0]

        # Модель windkessel.
        windkessel_model = WindkesselModel()
        windkessel_model.set_P(T, P)

        # Генератор синтетических данных.
        synhetic_artery = SyntheticArtery()

        # Задание параметров.
        synhetic_artery.T_max, synhetic_artery.T_s, synhetic_artery.T_d, \
                synhetic_artery.Q_max, synhetic_artery.R_f, windkessel_model.R, \
                windkessel_model.Z_0, windkessel_model.C, windkessel_model.P_out = x
        synhetic_artery.T = T[-1]

        # Синтетическое Q_in(t).
        synthetic_Q_in = numpy.array([synhetic_artery.get_Q_in(t) for t in T])
        windkessel_model.set_Q_in(T, synthetic_Q_in)

        # Синтетическое P(t).
        result = solve_ivp(lambda t, p : windkessel_model.P_rhs(t, p), (T[0], T[-1]), numpy.array([P[0]]),
                       t_eval=T, **solve_ivp_params)

        synthetic_P = result.y[0]

        value = numpy.sqrt(simps((P - synthetic_P)**2, T))
        return value


    def get_synthetic_artery_params(self, x0=None, bounds=None, n_points=100, scipy_minimize_params={"tol": 1e-7}, solve_ivp_params={}):
        """
        Получение параметров модели и синтетической артерии путём минимизации
        среднего квадрата отклонения измеренного и синтетического P(t).
        """

        if x0 is None:
            # TODO: убрать константы.
            x0 = numpy.array([0.09, 0.33, 0.35, 370.0, 0.3, self.R, self.Z_0, self.C, self.P_out])

        if bounds is None:
            # TODO: убрать константы.
            bounds = [
                (0.0, 0.2),
                (0.2, 0.4),
                (0.2, 0.4),
                (200.0, 2000.0),
                (0.0, 0.5),
                (0.2, 0.8),
                (0.02, 0.2),
                (0.1, 4.0),
                (0.0, 200.0)
            ]

        solve_ivp_params["rtol"] = 1.0
        solve_ivp_params["max_step"] = (self.T[-1] - self.T[0]) / n_points
        result = minimize(self.P_functional, x0=x0, args=(self.T, self.P, solve_ivp_params), bounds=bounds, **scipy_minimize_params)

        return result



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
