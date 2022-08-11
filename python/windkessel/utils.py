import numpy
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def calc_der(x, y):
    """
    Вычисление первой производной по времени.
    """

    der_y = numpy.zeros_like(y)

    # Производные на границах.
    der_y[0] = (y[1] - y[0]) / (x[1] - x[0])
    der_y[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

    # Производные внутри.
    der_y[1:-1] = 0.5 * ( (y[1:-1] - y[0:-2]) / (x[1:-1] - x[0:-2]) + (y[2:] - y[1:-1]) / (x[2:] - x[1:-1]) )

    return der_y



def loop_function(func, x_min, x_max):
    """
    Зацикливание одномерной функции.
    """

    return lambda x : func((x - x_min) % (x_max - x_min) + x_min)


def nonuniform_savgol_filter(x, y, n_points=None, calc_der=False, interp1d_params=None, savgol_filter_params=None):
    """
    Фильтр Савицки-Голая для неравномерной сетки.
    """

    # Сортировка.
    indexes = x.argsort()
    x = x[indexes]
    y = y[indexes]

    # Получение размера сетки, если не задан.
    length = x[-1] - x[0]
    if n_points is None:
        # Шаг будет не больше минимального шага по текущей сетке.
        delta = min(x[1:] - x[:-1])

        # Вычисляем число точек.
        n_points = int(numpy.ceil(length / delta))

    delta = length / n_points

    # Интерполяция.
    interpolant = interp1d(x, y, assume_sorted=True, **interp1d_params)
    uniform_x = numpy.linspace(x[0], x[-1], n_points)
    uniform_y = interpolant(uniform_x)

    # Сглаживание.
    smooth_y = savgol_filter(uniform_y, delta=delta, **savgol_filter_params)

    return uniform_x, smooth_y
