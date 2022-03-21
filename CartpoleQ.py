import itertools as it
import numpy as np
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from tqdm import tqdm

matplotlib.use("TKAgg")


class CartPole:
    def __init__(
        self,
        number_of_discret_values_states_params=None,
        number_of_discret_values_force=None,
    ):
        """
        Инициализация. Определение основных параметров маяника и начальных условий.
        Создание дискретных значений параметров среды. Задание правил игры.
        :param number_of_discret_values_states_params: int, количество дискретных значений параметров среды
        :param number_of_discret_values_force: int, количество дискретных значений внешнего воздействия
        """

        # physical constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force = 0.1
        self.tau = 0.02  # seconds between state updates

        # initial conditions
        self.initial_x = 0  # cart position
        self.initial_v = 0  # cart velocity
        self.initial_th = 0  # pendulum angle
        self.initial_w = 0  # pendulum angular velocity

        self.initial_state = np.array(
            [self.initial_th, self.initial_w, self.initial_x, self.initial_v]
        )

        # result of latest modeling
        self.latest_solution = None

        # for game conditions

        self.max_x = 2.4
        self.min_x = -2.4
        self.max_theta = 60 / 180 * np.pi
        self.min_theta = -60 / 180 * np.pi
        self.max_force = 10
        self.min_force = -10
        self.max_v = 10
        self.min_v = -10
        self.max_w = 10
        self.min_w = -10

        if number_of_discret_values_force is None:
            self.number_of_discret_values_force = 11
        else:
            self.number_of_discret_values_force = number_of_discret_values_force

        if number_of_discret_values_states_params is None:
            self.number_of_discret_values_x = 20
            self.number_of_discret_values_theta = 30
            self.number_of_discret_values_v = 40
            self.number_of_discret_values_w = 40
        else:
            self.number_of_discret_values_x = number_of_discret_values_states_params
            self.number_of_discret_values_theta = number_of_discret_values_states_params
            self.number_of_discret_values_v = number_of_discret_values_states_params
            self.number_of_discret_values_w = number_of_discret_values_states_params

        # bins
        self.bins_theta = np.linspace(
            self.min_theta, self.max_theta, self.number_of_discret_values_theta
        )
        self.bins_w = np.linspace(
            self.min_w, self.max_w, self.number_of_discret_values_w
        )
        self.bins_x = np.linspace(
            self.min_x, self.max_x, self.number_of_discret_values_x
        )
        self.bins_v = np.linspace(
            self.min_v, self.max_v, self.number_of_discret_values_v
        )
        self.list_of_bins = [self.bins_theta, self.bins_w, self.bins_x, self.bins_v]
        self.list_of_number_of_discret_values = [
            self.number_of_discret_values_theta,
            self.number_of_discret_values_w,
            self.number_of_discret_values_x,
            self.number_of_discret_values_v,
        ]

        # actions
        self.vailable_force = np.linspace(
            self.min_force, self.max_force, self.number_of_discret_values_force
        )

        # states
        self.list_of_discret_values = [
            [
                np.mean(bins[i: i + 2])
                for i in range(self.list_of_number_of_discret_values[k] - 1)
            ]
            for k, bins in enumerate(self.list_of_bins)
        ]

        # current info

        self.current_state = self.initial_state

    def _derivatives(self, state, time, force):
        """
        Функция определяет первые производные параметров среды для дальнейшего численного решения.
        :param state: np.array([theta, w, x, v]), текущее состояние среды
        :param time: np.array([]), длина которого определяет
        сколько шагов будет моделироваться маятник с текущего состояния
        :param force: float, величина силы, прикладываемой к тележке
        :return: np.array([theta, w, x, v]), вектор со значениями первых производных
        параметров текущего состояния среды
        """

        d_params = np.zeros_like(state)
        gravity, masspole, length = self.gravity, self.masspole, self.length
        total_mass, polemass_length = (
            self.total_mass,
            self.polemass_length,
        )

        d_params[0] = state[1]
        d_params[1] = (
            (
                gravity * np.sin(state[0])
                + np.cos(state[0])
                * (-force - polemass_length * state[1] ** 2 * np.sin(state[0]))
                / total_mass
            )
            / length
            / (4 / 3 - masspole * np.cos(state[0]) ** 2 / total_mass)
        )
        d_params[2] = state[3]
        d_params[3] = (
            force
            + polemass_length
            * (state[1] ** 2 * np.sin(state[0]) - d_params[1] * np.cos(state[0]))
        ) / total_mass

        return d_params

    def modeling_one_step(self, state=None, force=None):
        """
        Функция на основе текущего состояния среды и действия моделирует новое состояние среды
        :param state: np.array([theta, w, x, v]), текущее состояние среды
        :param force: float, величина силы, прикладываемой к тележке
        :return: np.array([theta, w, x, v]), новое состояние среды
        """

        if state is None:
            state = self.initial_state

        if force is None:
            force = self.force

        # integrate your ODE using scipy.integrate.
        new_state = integrate.odeint(
            self._derivatives, state, np.array([self.tau, 2 * self.tau]), args=(force,)
        )

        return new_state

    def modeling_with_q(self, q, state=None):
        """
        Функция, в которой Q-агент пытается удержать маятник вертикально.
        Агент наблюдает текущее состояние среды и выбирает, какое действие совершить.
        :param q: class, Q-агент, который принимает решения на основе Q-функции
        :param state: np.array([theta, w, x, v]), начальное состояние среды
        :return: dict{(state):[[q-value, action1], ..., [q-value, action_last]}, словарь, реализующий Q-table
                 np.array([[init state], ... ,[end state]]), массив со значениями всех смоделированных состояний среды
                 list, список за значениями наград на каждом шаге моделирования
                 list, список за сначениями совершённых действий на каждом шаге
        """

        if state is None:
            state = self.initial_state

        rewards = []
        actions = []

        all_states = state
        reward = self.get_reward(state)
        rewards.append(reward)

        while True:
            action = q.strategy(self.get_state(state))
            actions.append(action)
            force = self.vailable_force[action]
            new_state = self.modeling_one_step(state=state, force=force)[-1]
            all_states = np.vstack((all_states, new_state))

            if not self._game_on(new_state):
                break

            q.update_q_value(
                self.get_state(state), action, self.get_state(new_state), reward
            )

            self.current_state = new_state
            self.latest_solution = all_states
            state = new_state
            reward = self.get_reward(state)
            rewards.append(reward)

        return q.get_q_table(), all_states, rewards, actions

    def modeling_many_games(self, q, epochs=100):
        """
        Функция моделирует проигрывание Q-агентом заданного количества игр, в процессе чего обучается агент
        :param q: class, Q-агент, который принимает решения на основе Q-функции
        :param epochs: int, количества запланированных игр
        :return: class, Q-агент, который обучился принимать решения на основе Q-функции
                 list, список со значениями суммарных наград за всю игру
                 list, список со значениями шагов моделирования в процессе одной игры (длительность каждой игры)
                 list(list, ..., list), список, содержащий списки с совершёнными на каждом шаге в каждой игре действиями
        """

        len_of_epochs = []
        sum_rewards_of_epochs = []
        list_of_actions = []

        for i in tqdm(range(epochs)):
            q_table, _, rewards, actions = self.modeling_with_q(q)
            q.load_q_table(q_table)
            sum_rewards_of_epochs.append(np.sum(rewards))
            len_of_epochs.append(len(rewards))
            list_of_actions.append(actions)

        return q, sum_rewards_of_epochs, len_of_epochs, list_of_actions

    def _discretization_of_values(self, parameters):
        """
        Функция приводит полученные значения параметров среды к дискретизированным
        :param parameters: np.array([theta, w, x, v]), массив с недискретизированными параметрами среды
        :return: list, список с дискретизированными значениями параметров среды
        """

        list_of_number_of_discret_values = self.list_of_number_of_discret_values
        list_of_bins = self.list_of_bins
        list_of_discret_values = self.list_of_discret_values

        return [
            list_of_discret_values[k][i]
            for k, value in enumerate(parameters)
            for i in range(list_of_number_of_discret_values[k] - 1)
            if list_of_bins[k][i] < value < list_of_bins[k][i + 1]
        ]

    def states_and_actions(self):
        """
        Функция возвращает все возможные состояния среды и все возможные действия
        :return: list(tuple(state), ..., tuple(state),), список с кортежами параметров среды
                 list, список с действиями
        """

        states = list(it.product(*self.list_of_discret_values))
        actions = self.vailable_force

        return states, actions

    def get_reward(self, state=None):
        """
        Функция рассчитывает награду в текущем состоянии среды.
        Награда рассчитывается на основе дискретизированных значений.
        :param state: np.array([theta, w, x, v]), рассматриваемое состояние среды
        :return: float, награда в текущем состоянии среды
        """

        theta, w, x, v = self.get_state(state)

        reward = (
            1
            + np.e ** (np.abs(np.sin(theta)))
            - np.abs(x) / (self.max_x - self.min_x)
            - np.abs(w) / (self.max_w - self.min_w)
            - np.abs(v) / (self.max_v - self.min_v)
        )

        return reward

    def get_state(self, state=None):
        """
        Функция определяет текущее состояние среды и дискретизирует его.
        :param state: np.array([theta, w, x, v]), рассматриваемое состояние среды
        :return: tuple(), кортеж с дискретизированными параметрами среды
        """

        if state is None:
            theta, w, x, v = self.current_state
        else:
            theta, w, x, v = state

        theta, w, x, v = self._discretization_of_values([theta, w, x, v])

        return theta, w, x, v

    def _game_on(self, state):
        """
        Функция определяет, продолжается ли игра.
        :param state: np.array([theta, w, x, v]), рассматриваемое состояние среды
        :return: bool, True - игра продолжается, False - игра закончилас
        """

        theta, w, x, v = state
        valid = True

        if (
            (self.min_theta > theta)
            | (self.max_theta < theta)
            | (self.min_x > x)
            | (self.max_x < x)
        ):
            valid = False

        return valid

    def not_regulate_modeling(self, time=0, state=None, force=None):
        """
        Функция моделирует маятник, внешнее воздействие - постоянное.
        :param time: int, конечное время модуляции. Необходимо посмотреть шаг моделирования - tau.
        :param state: np.array([theta, w, x, v]), начальное состояние среды
        :param force: float, величина силы, прикладываемой к тележке
        :return: np.array([[init state], ... ,[end state]]), массив со значениями всех смоделированных состояний среды
        """

        tau = self.tau

        iterations = int((time + tau) / tau)

        solution = np.zeros((iterations, 4))

        if state is None:
            state = self.initial_state

        solution[0] = state

        for i in range(iterations - 1):
            solution[i + 1] = self.modeling_one_step(state=solution[i], force=force)[-1]

        self.latest_solution = solution
        self.current_state = solution[-1]

        return solution

    def plotting(self, file_name="free-cart.gif"):
        """
        Функция сохраняет в директории с файлом gif с последней сыгранной игрой. Смотреть latest_solution
        :param file_name: str, название файла с расширением gif
        """

        if self.latest_solution is None:
            raise Exception("Please do modeling, nothing to show")

        solution = self.latest_solution
        length = self.length
        dt = self.tau

        thetas = solution[:, 0]
        xs = solution[:, 2]

        pxs = length * np.sin(thetas) + xs
        pys = length * np.cos(thetas)

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.3, 5), ylim=(-1.2, 1.0))
        ax.set_aspect("equal")
        ax.grid()

        patch = ax.add_patch(
            Rectangle((0, 0), 0, 0, linewidth=1, edgecolor="k", facecolor="g")
        )

        (line,) = ax.plot([], [], "o-", lw=2)
        time_template = "time = %.1fs"
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        cart_width = 0.3
        cart_height = 0.2

        def init():
            line.set_data([], [])
            time_text.set_text("")
            patch.set_xy((-cart_width / 2, -cart_height / 2))
            patch.set_width(cart_width)
            patch.set_height(cart_height)
            return line, time_text, patch

        def animate(i):
            thisx = [xs[i], pxs[i]]
            thisy = [0, pys[i]]

            line.set_data(thisx, thisy)
            time_text.set_text(time_template % (i * dt))
            patch.set_x(xs[i] - cart_width / 2)
            return line, time_text, patch

        ani = animation.FuncAnimation(
            fig,
            animate,
            list(range(len(solution))),
            interval=25,
            blit=True,
            init_func=init,
        )

        writer_ = animation.writers["pillow"]
        writer = writer_(fps=25, bitrate=1800)
        ani.save(file_name, writer=writer)


class Qagent:
    def __init__(self):
        """
        Инициализация Q-агента. Задание параметров обучения.
        """

        self.gamma = 0.9
        self.alpha = 0.05
        self.epsilon = 0.1
        self.q_table = {}
        self.states = []
        self.actions = []

    def create_q_table(self, states, actions):
        """
        Функция создаёт q-table, с нулевыми значениями q.
        Реализация: dict{(state):[[q-value, action1], ..., [q-value, action_last]}.
        :param states: list(tuple(state), ..., tuple(state),), список с кортежами параметров среды
        :param actions: list, список с действиями
        """

        q_and_actions = [list(par) for par in zip(np.zeros(len(actions)), actions)]
        q_table = {state: q_and_actions for state in states}

        self.states = states
        self.actions = actions
        self.q_table = q_table

    def strategy(self, state):
        """
        Возвращает действие, выбранное агентом. Реализована eps-жадная стратегия.
        :param state: np.array([theta, w, x, v]), текущее состояние среды
        :return: int, индекс, выбранного агентом действия
        """

        q_table = self.q_table
        q_and_action = q_table[state]
        q_values = [q for q, action in q_and_action]
        actions = list(range(len(q_values)))

        if (np.random.rand() < self.epsilon) or (np.sum(q_values) == 0):
            action = np.random.choice(actions)
        else:
            action = actions[np.argmax(q_values)]

        return action

    def update_q_value(self, state, action, new_state, reward):
        """
        Функция обновляет значение q для выбранного действия.
        :param state: np.array([theta, w, x, v]), текущее состояние среды
        :param action: int, индекс, выбранного агентом действия
        :param new_state: np.array([theta, w, x, v]), следующее состояние среды,
        полученное из текущего состояния с помощью действия action
        :param reward: int, награда в текущем состоянии
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
        :param q_table: dict{(state):[[q-value, action1], ..., [q-value, action_last]}, словарь, реализующий Q-table
        """

        self.q_table = q_table

    def get_q_table(self):
        """
        Функция возвращает имеющуюся в классе Q-table.
        :return: dict{(state):[[q-value, action1], ..., [q-value, action_last]}, словарь, реализующий Q-table
        """

        return self.q_table
