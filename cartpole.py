import numpy as np
from space import Space


class CartPole:
    def __init__(
        self,
        frequency_of_discretization=None,
    ):
        """
        Инициализация. Определение основных параметров маяника и начальных условий.
        Создание дискретных значений параметров среды. Задание правил игры.
        frequency_of_discretization: int, количество дискретных значений параметров среды
        """

        # physical constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force = 10
        self.tau = 0.02  # seconds between state updates

        # initial conditions
        self.initial_x = 0  # cart position
        self.initial_v = 0  # cart velocity
        self.initial_th = 0  # pendulum angle
        self.initial_w = 0  # pendulum angular velocity

        # Angle at which to fail the episode
        self.theta_threshold_radians = 16 * 2 * np.pi / 360
        self.x_threshold = 2.4

        # max velocity and angle velocity
        self.max_v = 10
        self.max_w = 10

        # enviroment space bounds
        bound = np.array(
            [self.x_threshold, self.max_v, self.theta_threshold_radians, self.max_w]
        )

        # enviroment space
        self.env_space = Space(-bound, bound, frequency_of_discretization)
        self.states = self.env_space.get_states()

        # actions space
        self.action_space = Space(-self.force, self.force, 2)
        self.actions = self.action_space.get_states()

        # current state
        self.current_state = None

    def reset(self, random=True):
        """
        Функция инициализирует начальное значение среды. Случайно или в нуле
        :param random: bool, True - начальное значение определяется случайно, False - начальное значение равно 0
        :return: np.array([x, v, theta, w]), начальное состояние среды
        """

        env_space = self.env_space

        if random:
            initial_state = np.random.uniform(-0.05, 0.05, 4)

        else:
            initial_state = np.array(
                [self.initial_x, self.initial_v, self.initial_th, self.initial_w]
            )

        self.current_state = initial_state

        initial_state = env_space.discretization(initial_state)

        return initial_state

    def step(self, action):
        """
        Функция моделирует изменение состояния среды на один шаг, в соответствии с полученным действием
        :param action: int, индекс, выбранного агентом действия
        :return: np.array([theta, w, x, v]), новое состояние среды
                 int, награда в текущем состоянии среды
                 bool, True - игра окончена, False - игра продолжается
        """
        env_space = self.env_space
        current_state = self.current_state

        force = self.actions[action]
        new_state = self.update_state(current_state, force)
        observation = env_space.discretization(new_state)

        x, theta = new_state[::2]

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if done:
            reward = 0

        else:
            reward = 1

        self.current_state = new_state

        return observation, reward, done

    def update_state(self, state, force):
        """
        Функция определяет первые производные параметров среды для дальнейшего численного решения.
        state: np.array([x, v, theta, w]), текущее состояние среды
        force: float, величина силы, прикладываемой к тележке
        return: np.array([theta, w, x, v]), вектор со значениями первых производных
        параметров текущего состояния среды
        """

        tau = self.tau
        gravity, masspole, length, = (
            self.gravity,
            self.masspole,
            self.length,
        )
        total_mass, polemass_length = self.total_mass, self.polemass_length
        x_0, v_0, theta_0, w_0 = state

        sin = np.sin(theta_0)
        cos = np.cos(theta_0)

        dw = (
            (
                gravity * sin
                + cos * (-force - polemass_length * w_0**2 * sin) / total_mass
            )
            / length
            / (4 / 3 - masspole * cos**2 / total_mass)
        )

        dv = (force + polemass_length * (w_0**2 * sin - dw * cos)) / total_mass

        v_new = v_0 + dv * tau
        w_new = w_0 + dw * tau
        x_new = x_0 + v_0 * tau
        theta_new = theta_0 + w_0 * tau

        return [x_new, v_new, theta_new, w_new]

    def get_states_and_actions(self):
        """
        Функция возвращает все возможные состояния среды и все возможные действия
        :return: list(tuple(state), ..., tuple(state),), список с кортежами параметров среды
                 list, список с действиями
        """
        return self.states, self.actions
