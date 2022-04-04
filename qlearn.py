import numpy as np
#  from tqdm import tqdm


class Qagent:
    def __init__(self, bounds, num_env_discret_values, num_act_discret_values):
        """
        Инициализация Q-агента. Задание параметров обучения.
        bounds: np.array([x, v, theta, w]), текущее состояние среды
        num_env_discret_values: list or int, if list - количество значений каждого параметра среды,
                                            if int - одинаковое количество значений каждого параметра среды
        num_act_discret_values: int, количество действий
        """

        self.gamma = 0.95
        self.alpha = 0.8
        self.epsilon = 0.2
        self.q_table = {}
        self.lengths_games = []
        self.num_actions = num_act_discret_values

        if not type(bounds) is np.ndarray:
            bounds = np.array(bounds)

        if len(bounds.shape) == 1:
            bounds = np.array([-bounds, bounds])

        self.num_params_of_state = bounds.shape[1]

        if type(num_env_discret_values) is int:
            num_env_discret_values = [num_env_discret_values] * self.num_params_of_state

        self.discret_values = []

        for i in range(self.num_params_of_state):
            bins = np.linspace(bounds[0][i], bounds[1][i], num_env_discret_values[i])
            self.discret_values.append(bins)

        self.actions = np.array(range(num_act_discret_values))

    def fit(self, env, epochs, alpha=0.1, epsilon=1, gamma=0.85, inv_t=10, max_length_game=1000):

        '''
        Функция обучения агента
        env: среда, с которой взаимодействует агент
        epochs: int, количество игр
        alpha: float, скорость обучения
        epsilon: float, начальное значения epsilon
        gamma: float, коэффициент дисконтирования
        inv_t: float, коэффициент затухания epsilon
        max_length_game: int, максимальное количество игр
        '''

        self.lengths_games = []

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        for i_episode in range(epochs):  # tqdm(range(epochs)):

            t = 0

            observation = env.reset()
            self.epsilon *= np.exp(-i_episode / epochs * inv_t)

            while True:

                # if i_episode+1 == epochs:
                #    env.render()

                action = self.strategy(observation)
                new_observation, reward, done, _ = env.step(action)
                t += 1

                if done or (t > max_length_game):
                    self.lengths_games.append(t)
                    break

                self.update_q_value(observation, action, new_observation, reward + observation[2] * 2)
                observation = new_observation

            print(
                f'''Номер игры:{i_episode}, Медиана длительности игр:{np.median(self.lengths_games)}, 
                Мин.длит.игр:{np.min(self.lengths_games)}''',
                end='\r')

        # self.lengths_games = length_of_game
        # env.close()

    def strategy(self, state):
        """
        Возвращает действие, выбранное агентом. Реализована eps-жадная стратегия.
        state: np.array([x, v, theta, w]), текущее состояние среды
        return: int, индекс, выбранного агентом действия
        """

        q_table = self.q_table
        observation = tuple(self.get_observation(state))
        q_values = q_table.setdefault(observation)

        if q_values is None:
            q_values = np.zeros(self.num_actions)
            q_table[observation] = q_values

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
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
        observation = tuple(self.get_observation(state))
        new_observation = tuple(self.get_observation(new_state))

        q_values_of_new_state = q_table.setdefault(new_observation)

        if q_values_of_new_state is None:
            q_values_of_new_state = np.zeros(self.num_actions)
            q_table[new_observation] = q_values_of_new_state

        max_q_next = np.max(q_values_of_new_state)

        q_old = q_table[observation][action]
        q_table[observation][action] = q_old + self.alpha * (
                reward + self.gamma * max_q_next - q_old
        )
        self.q_table = q_table

    def get_observation(self, state):

        indices = []
        bins = self.discret_values

        for i in range(self.num_params_of_state):
            ind = np.digitize(state[i], bins[i])
            if ind == len(bins[i]):
                ind -= 1
            indices.append(ind)

        return indices

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
