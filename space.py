import itertools as it
import numpy as np


class Space:
    def __init__(self, low_thres, high_thres, frequency_of_discretization=20):
        """
        Инициализация класса, для работы с дискретизированным пространством
        low_thres: array, наименьшие значения параметров среды
        high_thres: array, наибольшие значения параметров среды
        frequency_of_discretization: int, частота дискретизации
        """

        if frequency_of_discretization != 2:
            delimeters_values = np.linspace(
                low_thres, high_thres, frequency_of_discretization, axis=0
            )
            discret_values = (delimeters_values[:-1] + delimeters_values[1:]) / 2

        else:
            delimeters_values = None
            discret_values = np.r_[low_thres, high_thres]

        self.low_thres = low_thres
        self.high_thres = high_thres
        self.frequency_of_discretization = frequency_of_discretization
        self.delimeters_values = delimeters_values
        self.discret_values = discret_values

    def discretization(self, state):
        """
        Функция дискретизирует входные значения
        state: array, недискретизированные значения параметров среды
        return: list, дискретизированные значения параметров среды
        """
        delimeters_values = self.delimeters_values
        discret_values = self.discret_values

        digitized_values = []

        for i in range(len(state)):
            ind_of_digit = np.digitize(state[i], delimeters_values.T[i])

            if ind_of_digit == 0:
                digitized_values.append(delimeters_values.T[i][0])

            elif ind_of_digit == len(delimeters_values):
                digitized_values.append(delimeters_values.T[i][-1])

            else:
                digitized_values.append(discret_values.T[i, ind_of_digit - 1])

        return digitized_values

    def get_states(self):
        """
        Функция возвращает все возможные состояния среды
        :return: list, список с кортежами параметров среды
        """
        discret_values = self.discret_values

        if len(discret_values.shape) == 1:
            states = discret_values
        else:
            states = list(it.product(*discret_values.T))
        return states

    def sample(self):
        """
        Получение случайных значений из дискретного пространства
        :return: int, индекс элемента выбранного случайно
        """
        return np.random.randint(len(self.discret_values))
