import numpy as np


class Qagent:
    def __init__(self):
        """
        Инициализация Q-агента. Задание параметров обучения.
        """

        self.gamma = 0.95
        self.alpha = 0.8
        self.epsilon = 0.2
        self.q_table = {}
        self.states = []
        self.actions = []

    def create_q_table(self, states, actions):
        """
        Функция создаёт q-table, с нулевыми значениями q.
        Реализация: dict{(state):[[q-value, action1], ..., [q-value, action_last]}.
        states: list(tuple(state), ..., tuple(state),), список с кортежами параметров среды
        actions: list, список с действиями
        """

        q_and_actions = [list(par) for par in zip(np.zeros(len(actions)), actions)]
        q_table = {state: q_and_actions for state in states}

        self.states = states
        self.actions = actions
        self.q_table = q_table

    def strategy(self, state):
        """
        Возвращает действие, выбранное агентом. Реализована eps-жадная стратегия.
        state: np.array([x, v, theta, w]), текущее состояние среды
        return: int, индекс, выбранного агентом действия
        """

        q_table = self.q_table
        q_and_action = q_table[state]
        q_values = [q for q, action in q_and_action]
        actions = list(range(len(q_values)))

        if (np.random.rand() < self.epsilon) or (np.sum(q_values) == 0):
            action = np.random.choice(actions)
        else:
            action = np.argmax(q_values)

        return action

    def update_q_value(self, state, action, new_state, reward):
        """
        Функция обновляет значение q для выбранного действия.
        state: np.array([x, v, theta, w]), текущее состояние среды
        action: int, индекс, выбранного агентом действия
        new_state: np.array([x, v, theta, w]), следующее состояние среды,
        полученное из текущего состояния с помощью действия action
        reward: int, награда в текущем состоянии
        """

        q_table = self.q_table
        q_values_of_new_state = [q for q, action in q_table[new_state]]
        max_q_next = np.max(q_values_of_new_state)
        q_old = q_table[state][action][0]
        q_table[state][action][0] = q_old + self.alpha * (
            reward + self.gamma * max_q_next - q_old
        )
        self.q_table = q_table

    def load_q_table(self, q_table):
        """
        Функция загружает в класс, уже имеющуюся Q-table.
        q_table: dict{(state):[[q-value, action1], ..., [q-value, action_last]}, словарь, реализующий Q-table
        """

        self.q_table = q_table

    def get_q_table(self):
        """
        Функция возвращает имеющуюся в классе Q-table.
        return: dict{(state):[[q-value, action1], ..., [q-value, action_last]}, словарь, реализующий Q-table
        """

        return self.q_table
