from abc import ABC, abstractmethod
import numpy as np
import sympy as sym
from sympy import Piecewise
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline as SciPyBSpline, CubicSpline as SciPyCubicSpline
from typing import Callable, List
from functools import lru_cache
from scipy.linalg import solve


@lru_cache
def factorial(n):
	if n <= 1:
		return 1

	return n * factorial(n - 1)


class Spline(ABC):
	"""Базовый абстрактный класс для всех типов сплайнов"""

	@abstractmethod
	def fit(self, x: np.ndarray, y: np.ndarray) -> None:
		"""Обучение сплайна на данных"""
		pass

	@abstractmethod
	def predict(self, x: np.ndarray) -> np.ndarray:
		"""Предсказание значений сплайна в точках x"""
		pass

	@abstractmethod
	def get_basis_functions(self) -> List[Callable]:
		"""Получение базисных функций сплайна"""
		pass


class PTSpline(Spline):
	"""Penalized Truncated Power Basis Spline"""

	def __init__(self, degree: int, knots: np.ndarray, smoothing_param: float = 1.0):
		"""
		Инициализация PT-сплайна

		Args:
			degree: Степень полинома
			knots: Узлы сплайна
			smoothing_param: Параметр сглаживания
		"""
		self.degree = degree
		self.knots = knots
		self.smoothing_param = smoothing_param
		self.coefficients = None
		self.basis_functions = None

	def truncated_power_basis(self, x: np.ndarray) -> np.ndarray:
		"""Создание базиса усеченных степенных функций"""
		# Полиномиальная часть: 1, x, x^2, ..., x^p
		X_poly = np.vstack([x ** d for d in range(self.degree + 1)]).T

		# Усеченные функции: (x - κ)_+^p
		X_trunc = np.vstack([np.where(x > knot, (x - knot) ** self.degree, 0)
							 for knot in self.knots]).T

		# Объединяем обе части
		X = np.hstack([X_poly, X_trunc])
		return X

	def fit(self, x: np.ndarray, y: np.ndarray) -> None:
		"""Обучение PT-сплайна"""
		if self.knots is None:
			# Автоматическое определение узлов
			num_knots = max(int(len(x) / 4), 4)
			self.knots = np.linspace(min(x), max(x), num_knots)

		X = self.truncated_power_basis(x)

		# Создаем матрицу D: для полиномиальной части (первые p+1 коэффициентов штраф не применяется)
		D = np.diag([0] * (self.degree + 1) + [1] * len(self.knots))

		# Штрафной множитель λ^(2p)
		penalty = self.smoothing_param ** (2 * self.degree)

		# Решаем систему
		A = X.T @ X + penalty * D
		b = X.T @ y
		self.coefficients = np.linalg.lstsq(A, b, rcond=None)[0]

		# Сохраняем базисные функции
		self.basis_functions = X

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""Предсказание значений"""
		if self.coefficients is None:
			raise ValueError("Spline not fitted yet")
		X_new = self.truncated_power_basis(x)
		return X_new @ self.coefficients

	def get_basis_functions(self) -> List[Callable]:
		"""Получение базисных функций"""
		return self.basis_functions

	@staticmethod
	def demo():
		"""Визуальный тест PTSpline с разными значениями lambda"""

		np.random.seed(42)
		x = np.linspace(0, 2 * np.pi, 250)
		x_dense = np.linspace(0, 2 * np.pi, 200)

		# Истинные функции
		y_sin_true = np.sin(5 * x)
		y_sin_true_dense = np.sin(5 * x_dense)
		y_sin_noisy = y_sin_true + np.random.normal(0, 0.2, len(x))

		y_poly_true = 0.5 * x ** 2 - x + 1
		y_poly_true_dense = 0.5 * x_dense ** 2 - x_dense + 1
		y_poly_noisy = y_poly_true + np.random.normal(0, 0.3, len(x))

		# Параметры
		lambdas = [0.1, 1.0]
		colors = ['green', 'purple']
		degree = 3
		knots = np.linspace(0, 2 * np.pi, 10)

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

		# Аппроксимация синуса
		ax1.scatter(x, y_sin_noisy, alpha=0.6, color='blue', label='Данные с шумом', s=30)
		ax1.plot(x_dense, y_sin_true_dense, 'r--', label='Истинная функция', linewidth=3)

		for lambda_val, color in zip(lambdas, colors):
			pt_sin = PTSpline(degree=degree, knots=knots, smoothing_param=lambda_val)
			pt_sin.fit(x, y_sin_noisy)
			y_sin_pred = pt_sin.predict(x_dense)
			ax1.plot(x_dense, y_sin_pred, color=color, label=f'PTSpline (λ={lambda_val})', linewidth=2)

		ax1.set_title('Аппроксимация синуса', fontsize=14)
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.legend()
		ax1.grid(True, alpha=0.3)

		# Аппроксимация полинома
		ax2.scatter(x, y_poly_noisy, alpha=0.6, color='blue', label='Данные с шумом', s=30)
		ax2.plot(x_dense, y_poly_true_dense, 'r--', label='Истинная функция', linewidth=3)

		for lambda_val, color in zip(lambdas, colors):
			pt_poly = PTSpline(degree=degree, knots=knots, smoothing_param=lambda_val)
			pt_poly.fit(x, y_poly_noisy)
			y_poly_pred = pt_poly.predict(x_dense)
			ax2.plot(x_dense, y_poly_pred, color=color, label=f'PTSpline (λ={lambda_val})', linewidth=2)

		ax2.set_title('Аппроксимация полинома', fontsize=14)
		ax2.set_xlabel('x')
		ax2.set_ylabel('y')
		ax2.legend()
		ax2.grid(True, alpha=0.3)

		plt.suptitle('Тестирование PTSpline с разными λ', fontsize=16)
		plt.tight_layout()
		plt.show()

	@staticmethod
	def plot(x, y, degree=3, smoothing_param=1.0, knots=None,
			 show_data=True, color='blue', title=None,
			 num_points=300, figsize=(10, 6), grid=True, legend=True):
		"""
		Визуализация Penalized Truncated Spline по заданным параметрам.

		Параметры:
		- x, y: данные
		- degree: степень полинома
		- smoothing_param: параметр сглаживания λ
		- knots: массив внутренних узлов или None
		- show_data: отображать ли точки
		- color: цвет линии сплайна
		- title: заголовок графика
		- num_points: число точек на графике
		- figsize: размер фигуры
		- grid, legend: отображение сетки и легенды
		"""
		x = np.asarray(x)
		y = np.asarray(y)

		if knots is None:
			num_knots = max(int(len(x) / 4), 4)
			knots = np.linspace(min(x), max(x), num_knots)

		spline = PTSpline(degree=degree, knots=knots, smoothing_param=smoothing_param)
		spline.fit(x, y)

		x_dense = np.linspace(min(x), max(x), num_points)
		y_pred = spline.predict(x_dense)

		plt.figure(figsize=figsize)
		if show_data:
			plt.scatter(x, y, color='red', alpha=0.6, s=20, label='Данные')
		plt.plot(x_dense, y_pred, color=color, lw=2, label=f'PTSpline (λ={smoothing_param})')

		plt.xlabel("x")
		plt.ylabel("y")
		plt.title(title or "PTSpline")
		if grid:
			plt.grid(True)
		if legend:
			plt.legend()
		plt.tight_layout()
		plt.show()


class PSpline(Spline):
	def __init__(self, degree=3, knots=None, penalty_order=2, lambda_=1.0):
		self.degree = degree
		self.knots = knots
		self.penalty_order = penalty_order
		self.lambda_ = lambda_
		self.coefficients = None
		self.basis_functions = None
		self.spline = None
		self.boundary_conditions = None

	def _difference_matrix(self, n_bases, d):
		"""Создает разностную матрицу порядка d."""
		D = np.eye(n_bases)
		for _ in range(d):
			D = np.diff(D, n=1, axis=0)
		return D

	def set_boundary_conditions(self, bc_type, bc_values=None):
		"""Задает граничные условия для сплайна."""
		if bc_type not in [None, 'natural', 'clamped', 'cyclic']:
			raise ValueError(
				"Поддерживаемые типы граничных условий: 'natural', 'clamped', 'cyclic'.")

		if bc_type == 'clamped' and (bc_values is None or 'left' not in bc_values or 'right' not in bc_values):
			raise ValueError(
				"Для 'clamped' граничных условий необходимо предоставить 'left' и 'right' значения производных.")

		self.boundary_conditions = {
			'type': bc_type,
			'values': bc_values
		}

	def fit(self, x, y, penalty_fun=None):
		"""Аппроксимирует P-сплайн к данным с учетом функции штрафа."""
		if self.knots is None:
			num_internal_knots = max(int(len(x) / 4), 4)
			self.knots = np.linspace(min(x), max(x), num_internal_knots)
			self.knots = np.concatenate((
				[x[0]] * self.degree,
				self.knots,
				[x[-1]] * self.degree
			))

		n = len(x)
		t = self.knots
		k = self.degree
		n_bases = len(t) - k - 1

		# Создаем базисную матрицу B
		B = np.zeros((n, n_bases))
		for i in range(n_bases):
			c = np.zeros(n_bases)
			c[i] = 1
			spline = SciPyBSpline(t, c, k)
			B[:, i] = spline(x)

		# Создаем разностную матрицу
		D = self._difference_matrix(n_bases, self.penalty_order)

		# Применяем функтор к разностной матрице, если он задан
		if penalty_fun is not None:
			D = penalty_fun(D)
			D = np.abs(D)

		# Создаем штрафную матрицу P
		P = self.lambda_ * D.T @ D

		# Основная система уравнений: (B^T B + P) c = B^T y
		BtB = B.T @ B
		Bty = B.T @ y
		A = BtB + P
		rhs = Bty.copy()

		# Обработка граничных условий
		if self.boundary_conditions is not None:
			bc_type = self.boundary_conditions['type']
			if bc_type == 'natural':
				# Вторая производная на концах равна нулю
				B_der2_left = np.array([
					SciPyBSpline(t, np.eye(n_bases)[i], k).derivative(2)(x[0])
					for i in range(n_bases)
				])
				B_der2_right = np.array([
					SciPyBSpline(t, np.eye(n_bases)[i], k).derivative(2)(x[-1])
					for i in range(n_bases)
				])
				A = np.vstack([A, B_der2_left, B_der2_right])
				rhs = np.hstack([rhs, 0, 0])

			elif bc_type == 'clamped':
				bc_values = self.boundary_conditions['values']
				B_der1_left = np.array([
					SciPyBSpline(t, np.eye(n_bases)[i], k).derivative(1)(x[0])
					for i in range(n_bases)
				])
				B_der1_right = np.array([
					SciPyBSpline(t, np.eye(n_bases)[i], k).derivative(1)(x[-1])
					for i in range(n_bases)
				])
				A = np.vstack([A, B_der1_left, B_der1_right])
				rhs = np.hstack([rhs, bc_values['left'], bc_values['right']])

			elif bc_type == 'cyclic':
				self._set_cyclic_boundary_conditions(A, rhs, x, t, k, n_bases)

		# Решаем систему уравнений
		try:
			self.coefficients = np.linalg.lstsq(A, rhs, rcond=None)[0]
		except np.linalg.LinAlgError as e:
			raise np.linalg.LinAlgError(
				f"Ошибка при решении системы уравнений: {e}")

		self.spline = SciPyBSpline(self.knots, self.coefficients, self.degree)
		self.basis_functions = B

	def _set_cyclic_boundary_conditions(self, A, rhs, x, t, k, n_bases):
		"""Задает циклические граничные условия для сплайна."""
		# Условия на совпадение значений сплайна на концах
		B_start = np.array([
			SciPyBSpline(t, np.eye(n_bases)[i], k)(x[0])
			for i in range(n_bases)
		])
		B_end = np.array([
			SciPyBSpline(t, np.eye(n_bases)[i], k)(x[-1])
			for i in range(n_bases)
		])
		continuity_row = B_start - B_end

		# Условия на совпадение первой производной
		B_der1_start = np.array([
			SciPyBSpline(t, np.eye(n_bases)[i], k).derivative(1)(x[0])
			for i in range(n_bases)
		])
		B_der1_end = np.array([
			SciPyBSpline(t, np.eye(n_bases)[i], k).derivative(1)(x[-1])
			for i in range(n_bases)
		])
		derivative1_row = B_der1_start - B_der1_end

		# Нормализация и увеличение веса условий
		weight = 1e3
		continuity_row = continuity_row / \
						 np.linalg.norm(continuity_row) * weight
		derivative1_row = derivative1_row / \
						  np.linalg.norm(derivative1_row) * weight

		A = np.vstack([A, continuity_row, derivative1_row])
		rhs = np.hstack([rhs, 0, 0])

	def predict(self, x):
		"""Предсказывает значения y для новых значений x."""
		if self.spline is None:
			raise ValueError("Сначала нужно выполнить fit")
		return self.spline(x)

	def get_basis_functions(self):
		"""Возвращает базисные функции сплайна."""
		return self.basis_functions

	def plot(self, x_range=None, num_points=100):
		"""Построение графика сплайна вместе с исходными данными."""
		if x_range is None:
			x_range = (min(self.knots), max(self.knots))

		x_vals = np.linspace(x_range[0], x_range[1], num_points)
		y_vals = self.predict(x_vals)

		plt.figure(figsize=(10, 6))
		plt.plot(x_vals, y_vals, label=f"{self.__class__.__name__} сплайн")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title(f"Построение {self.__class__.__name__} сплайна")
		plt.legend()
		plt.grid(True)
		plt.show()

	@staticmethod
	def demo():
		"""Демонстрация: аппроксимация шумных данных с помощью P-сплайна"""
		np.random.seed(42)
		x = np.linspace(0, 10, 100)
		y_true = 0.5 * x + np.sin(x)
		y = y_true + np.random.normal(scale=0.5, size=x.shape)

		pspline = PSpline(degree=3, penalty_order=2, lambda_=1.0)
		pspline.fit(x, y)

		x_range = (min(x), max(x))
		x_vals = np.linspace(x_range[0], x_range[1], 200)
		y_vals = pspline.predict(x_vals)

		plt.figure(figsize=(10, 6))
		plt.plot(x_vals, y_vals, label=f"{pspline.__class__.__name__} сплайн")
		plt.scatter(x, y, color='red', s=20, alpha=0.7, label="Данные")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title(f"Построение {pspline.__class__.__name__} сплайна")
		plt.legend()
		plt.grid(True)
		plt.show()

	@staticmethod
	def plot(x, y, degree=3, penalty_order=2, lambda_=1.0,
			 knots=None, show_data=True, color='blue', title=None,
			 num_points=300, figsize=(10, 6), grid=True, legend=True):
		"""
		Визуализация P-сплайна с параметрами.
		"""
		spline = PSpline(degree=degree, penalty_order=penalty_order, lambda_=lambda_, knots=knots)
		spline.fit(x, y)

		x_dense = np.linspace(min(x), max(x), num_points)
		y_pred = spline.predict(x_dense)

		plt.figure(figsize=figsize)
		if show_data:
			plt.scatter(x, y, color='red', alpha=0.6, label='Данные', s=20)
		plt.plot(x_dense, y_pred, color=color, label=f'P-сплайн λ={lambda_}')
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title(title or "P-сплайн")
		if grid:
			plt.grid(True)
		if legend:
			plt.legend()
		plt.show()


class SplineSegment:
	def __init__(self, x, y):
		self.x = x  # узловое значение x
		self.y = y  # наблюдаемое значение y (из данных)
		self.a = 0.0  # коэффициент при (x - x_i)^3
		self.b = 0.0  # коэффициент при (x - x_i)^2
		self.c = 0.0  # коэффициент при (x - x_i)
		self.d = 0.0  # свободный член (значение функции в узле)


class SmoothingCubicSpline(Spline):
	"""
	Сглаживающий кубический сплайн
	"""

	def __init__(self, lam: float, sigma: np.ndarray = None, m0: float = None, mn: float = None):
		self.segments = []
		self.lam = max(0, min(1, lam))  # NOTE(kon3gor): Clamp lambda value to [0, 1]
		self.sigma = sigma
		self.m0 = m0
		self.mn = mn
		self.is_clamped = self.m0 is not None and self.mn is not None

	def fit(self, x: np.ndarray, y: np.ndarray) -> None:
		self.sigma = self.sigma if self.sigma is not None else np.ones_like(x)
		self.sigma = np.asarray(self.sigma)

		if len(x) != len(y) or len(x) != len(self.sigma):
			raise ValueError("Data points should have the same dimension as sigma array")

		if len(x) < 3:
			raise ValueError("There should be at least 3 knots for this spline to be constructed")

		if self.is_clamped:
			return self.__fit_clamped(x, y)

		return self.__fit_regular(x, y)

	def __fit_clamped(self, x: np.ndarray, y: np.ndarray) -> None:
		# Интерполяционный случай
		if self.lam >= 1.0 - 1e-12:
			self.cs = SciPyCubicSpline(x, y, bc_type=((1, self.m0), (1, self.mn)))
			return

		# Сглаживающий случай
		W = np.diag(1 / self.sigma ** 2)
		h = np.diff(x)
		n = len(x)
		N = n

		# Строим матрицу гладкости (аналогично spline smoothing)
		D = np.zeros((N, N))
		for i in range(1, N - 1):
			D[i, i - 1] = 1 / h[i - 1]
			D[i, i] = -1 / h[i - 1] - 1 / h[i]
			D[i, i + 1] = 1 / h[i]
		# Граничные условия: закреплённые производные
		D[0, 0] = 1
		D[-1, -1] = 1

		A = W + self.lam * D.T @ D

		# Матрица ограничений на производные
		C = np.zeros((2, len(x)))
		C[0, :2] = [-1 / (x[1] - x[0]), 1 / (x[1] - x[0])]  # ≈ f'(x0)
		C[1, -2:] = [-1 / (x[-1] - x[-2]), 1 / (x[-1] - x[-2])]  # ≈ f'(xn)

		b = np.array([m0, mn])

		# Формируем расширенную KKT-систему
		KKT = np.block([
			[A, C.T],
			[C, np.zeros((2, 2))]
		])
		rhs = np.concatenate([W @ y, b])

		# Решаем KKT
		sol = solve(KKT, rhs)
		y_smooth = sol[:len(x)]

		self.cs = SciPyCubicSpline(x, y_smooth, bc_type=((1, self.m0), (1, self.mn)))

	def __fit_regular(self, x: np.ndarray, y: np.ndarray) -> None:
		"""Обучение сплайна на данных"""

		N = len(x)
		n = N - 1  # число сегментов

		# Создаем массив объектов SplineSegment, где для каждого узла заданы x и y.
		S = [SplineSegment(x[i], y[i]) for i in range(N)]

		# Инициализируем вспомогательные массивы.
		h = np.zeros(n)  # h[i] = x[i+1] - x[i]
		r = np.zeros(n + 1)  # r[i] = 3/h[i] для i от 0 до n-1, r[n] = 0
		f = np.zeros(n + 1)  # f[i] = - (r[i-1] + r[i])
		p = np.zeros(n + 1)  # p[i] = 2*(x[i+1] - x[i-1])
		# q[i] = 3*(y[i+1]-y[i])/h[i] - 3*(y[i]-y[i-1])/h[i-1]
		q = np.zeros(n + 1)
		u = np.zeros(n + 2)  # для решения системы, длина n+2
		v = np.zeros(n + 2)
		w = np.zeros(n + 2)

		# Первый сегмент
		h[0] = S[1].x - S[0].x
		r[0] = 3.0 / h[0]

		# Вычисляем h, r, f, p, q для внутренних узлов (i = 1 .. n-1)
		for i in range(1, n):
			h[i] = S[i + 1].x - S[i].x
			r[i] = 3.0 / h[i]
			f[i] = - (r[i - 1] + r[i])
			p[i] = 2.0 * (S[i + 1].x - S[i - 1].x)
			q[i] = 3.0 * (S[i + 1].y - S[i].y) / h[i] - \
				   3.0 * (S[i].y - S[i - 1].y) / h[i - 1]
		r[n] = 0.0  # крайнее условие
		f[0] = 0.0
		f[n] = 0.0

		# Параметр mu связывает вклад сглаживания и приближения данных.
		mu = 2.0 * (1.0 - self.lam) / (3.0 * self.lam)

		# Вычисляем u, v, w для i = 1 .. n-1
		for i in range(1, n):
			u[i] = (r[i - 1] ** 2 * self.sigma[i - 1] +
					f[i] ** 2 * self.sigma[i] +
					r[i] ** 2 * self.sigma[i + 1])
			u[i] = mu * u[i] + p[i]
			v[i] = f[i] * r[i] * self.sigma[i] + r[i] * \
				   (f[i + 1] if i + 1 < len(f) else 0.0) * self.sigma[i + 1]
			v[i] = mu * v[i] + h[i]
			w[i] = mu * r[i] * (r[i + 1] if i + 1 < len(r) else 0.0) * self.sigma[i + 1]

		# Задаем граничные элементы (индексы 0, n, n+1)
		u[0] = 0.0
		u[n] = 0.0
		u[n + 1] = 0.0
		v[0] = 0.0
		v[n] = 0.0
		v[n + 1] = 0.0
		w[0] = 0.0
		w[n] = 0.0
		w[n + 1] = 0.0

		# Копируем q в массив Q длины n+2, оставляя граничные элементы нулевыми.
		Q = np.zeros(n + 2)
		Q[1:n] = q[1:n]

		# Решаем систему методом Quincunx для Q (индексы 1..n-1)
		self.__quincunx(u, v, w, Q, n)

		# Восстанавливаем параметры сплайна.
		S[0].d = S[0].y - mu * r[0] * Q[1] * self.sigma[0]
		if n > 0:
			S[1].d = S[1].y - mu * (f[1] * Q[1] + r[1] * Q[2]) * self.sigma[1]
			S[0].a = Q[1] / (3.0 * h[0])
			S[0].b = 0.0
			S[0].c = (S[1].d - S[0].d) / h[0] - Q[1] * h[0] / 3.0
		r[0] = 0.0

		for j in range(1, n):
			S[j].a = (Q[j + 1] - Q[j]) / (3.0 * h[j])
			S[j].b = Q[j]
			S[j].c = (Q[j] + Q[j - 1]) * h[j - 1] + S[j - 1].c
			temp = (r[j - 1] * Q[j - 1] + f[j] * Q[j] + r[j] * Q[j + 1])
			S[j].d = S[j].y - mu * temp * self.sigma[j]

		S[-1].d = S[-1].y
		S[-1].a = 0.0
		S[-1].b = 0.0
		S[-1].c = 0.0

		self.segments = S

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""Предсказание значений сплайна в точках x"""
		if self.is_clamped:
			return self.cs(x)

		y_out = []
		for x_i in x:
			i = 0
			while i < len(self.segments) and self.segments[i].x <= x_i:
				i += 1

			if i == len(self.segments) and self.segments[-1].x <= x_i:
				continue

			seg = self.segments[i - 1]
			y = seg.a * (x_i - seg.x) ** 3 + seg.b * (x_i - seg.x) ** 2 + seg.c * (x_i - seg.x) + seg.d
			y_out.append(y)

		return np.asarray(y_out)

	def get_basis_functions(self) -> List[Callable]:
		"""Получение базисных функций сплайна"""
		if not self.is_clamped:
			return []

		basis = []
		for i in range(len(self.x) - 1):
			def f(x, i=i):
				return self.cs(x) * (self.x[i] <= x) * (x <= self.x[i + 1])

			basis.append(f)

		return basis

	def __quincunx(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, q: np.ndarray, n: np.ndarray) -> None:
		"""
		Решает пятидиагональную систему методом Quincunx.
		Массивы u, v, w, q имеют длину n+2, решения ищутся для индексов 1..n-1.
		"""
		# Факторизация и прямой ход
		for j in range(1, n):
			term1 = u[j - 2] * (w[j - 2] ** 2) if j - 2 >= 1 else 0.0
			term2 = u[j - 1] * (v[j - 1] ** 2) if j - 1 >= 1 else 0.0
			u[j] = u[j] - term1 - term2
			if u[j] == 0:
				raise ValueError("Нулевой делитель в факторизации Quincunx")
			v[j] = (v[j] - (u[j - 1] * v[j - 1] * w[j - 1] if j - 1 >= 1 else 0.0)) / u[j]
			w[j] = w[j] / u[j]

		# Прямой ход (подстановка)
		for j in range(1, n):
			term1 = v[j - 1] * q[j - 1] if j - 1 >= 1 else 0.0
			term2 = w[j - 2] * q[j - 2] if j - 2 >= 1 else 0.0
			q[j] = q[j] - term1 - term2
		for j in range(1, n):
			q[j] = q[j] / u[j]

		# Обратный ход
		q[n + 1] = 0.0
		q[n] = 0.0
		for j in range(n - 1, 0, -1):
			q[j] = q[j] - v[j] * q[j + 1] - w[j] * q[j + 2]

	@staticmethod
	def demo(noise_level: float = 0.8,
			 lambdas: List[float] = [0.0001, 0.25, 0.5, 0.75, 1],
			 colors: List[str] = ["r", "g", "b", "magenta", "yellow"],
			 show_data: bool = True,
			 grid: bool = True,
			 legend: bool = True,
			 figsize: tuple = (10, 6),
			 title: str = 'Сглаживающий сплайн для cos(x)'):
		"""
		Демонстрация сглаживающего кубического сплайна с различными λ.
		"""

		np.random.seed(42)
		x = np.linspace(0, 10, 20)
		y = np.cos(x)
		y_noisy = y + noise_level * np.random.randn(len(x))
		sigma = np.ones_like(x)

		plt.figure(figsize=figsize)

		if show_data:
			plt.plot(x, y_noisy, 'o', label='Исходные данные')

		for lam, color in zip(lambdas, colors):
			spline = SmoothingCubicSpline(lam=lam, sigma=sigma)
			spline.fit(x, y_noisy)
			xx = np.linspace(min(x), max(x), 1000)
			y_pred = spline.predict(xx)
			plt.plot(xx[:-1], y_pred, color, lw=2, label=f'λ={lam}')

		plt.xlabel('x')
		plt.ylabel('y')
		plt.title(title)
		if grid:
			plt.grid(True)
		if legend:
			plt.legend()
		plt.show()


class CubicClosedSpline(Spline):
	"""
	Сглаживающий кубический сплайн по алгоритму из smoothing.ipynb (Pollock, 1999),
	с интерфейсом fit, predict, get_basis_functions.
	"""

	def __init__(self):
		self.x = None
		self.y = None
		self.sigma = None
		self.lam = None
		self.segments = None  # Список сегментов с коэффициентами a, b, c, d

	def fit(self, x, y, sigma=None, lam=1.0):
		"""
		Обучение сглаживающего кубического сплайна.
		x, y - данные
		sigma - веса (стандартные отклонения), если None, то все веса = 1
		lam - параметр сглаживания (0 < lam <= 1)
		"""
		x = np.asarray(x)
		y = np.asarray(y)
		N = len(x)
		if N < 3:
			raise ValueError(
				"Должно быть не менее 3 узлов для построения сплайна.")
		if sigma is None:
			sigma = np.ones_like(x)
		else:
			sigma = np.asarray(sigma)
		n = N - 1
		# Сохраняем параметры
		self.x = x
		self.y = y
		self.sigma = sigma
		self.lam = lam

		# Структура сегмента

		class SplineSegment:
			def __init__(self, x, y):
				self.x = x  # узловое значение x
				self.y = y  # наблюдаемое значение y (из данных)
				self.a = 0.0  # коэффициент при (x - x_i)^3
				self.b = 0.0  # коэффициент при (x - x_i)^2
				self.c = 0.0  # коэффициент при (x - x_i)
				self.d = 0.0  # свободный член (значение функции в узле)

		S = [SplineSegment(x[i], y[i]) for i in range(N)]
		h = np.zeros(n)
		r = np.zeros(n + 1)
		f = np.zeros(n + 1)
		p = np.zeros(n + 1)
		q = np.zeros(n + 1)
		u = np.zeros(n + 2)
		v = np.zeros(n + 2)
		w = np.zeros(n + 2)
		h[0] = S[1].x - S[0].x
		r[0] = 3.0 / h[0]
		for i in range(1, n):
			h[i] = S[i + 1].x - S[i].x
			r[i] = 3.0 / h[i]
			f[i] = - (r[i - 1] + r[i])
			p[i] = 2.0 * (S[i + 1].x - S[i - 1].x)
			q[i] = 3.0 * (S[i + 1].y - S[i].y) / h[i] - \
				   3.0 * (S[i].y - S[i - 1].y) / h[i - 1]
		r[n] = 0.0
		f[0] = 0.0
		f[n] = 0.0
		mu = 2.0 * (1.0 - lam) / (3.0 * lam)
		for i in range(1, n):
			u[i] = (r[i - 1] ** 2 * sigma[i - 1] +
					f[i] ** 2 * sigma[i] +
					r[i] ** 2 * sigma[i + 1])
			u[i] = mu * u[i] + p[i]
			v[i] = f[i] * r[i] * sigma[i] + r[i] * \
				   (f[i + 1] if i + 1 < len(f) else 0.0) * sigma[i + 1]
			v[i] = mu * v[i] + h[i]
			w[i] = mu * r[i] * (r[i + 1] if i + 1 < len(r) else 0.0) * sigma[i + 1]
		u[0] = 0.0
		u[n] = 0.0
		u[n + 1] = 0.0
		v[0] = 0.0
		v[n] = 0.0
		v[n + 1] = 0.0
		w[0] = 0.0
		w[n] = 0.0
		w[n + 1] = 0.0
		Q = np.zeros(n + 2)
		Q[1:n] = q[1:n]

		# Решение пятидиагональной системы (метод quincunx)

		def quincunx(u, v, w, q, n):
			for j in range(1, n):
				term1 = u[j - 2] * (w[j - 2] ** 2) if j - 2 >= 1 else 0.0
				term2 = u[j - 1] * (v[j - 1] ** 2) if j - 1 >= 1 else 0.0
				u[j] = u[j] - term1 - term2
				if u[j] == 0:
					raise ValueError(
						"Нулевой делитель в факторизации Quincunx")
				v[j] = (v[j] - (u[j - 1] * v[j - 1] * w[j - 1]
								if j - 1 >= 1 else 0.0)) / u[j]
				w[j] = w[j] / u[j]
			for j in range(1, n):
				term1 = v[j - 1] * q[j - 1] if j - 1 >= 1 else 0.0
				term2 = w[j - 2] * q[j - 2] if j - 2 >= 1 else 0.0
				q[j] = q[j] - term1 - term2
			for j in range(1, n):
				q[j] = q[j] / u[j]
			q[n + 1] = 0.0
			q[n] = 0.0
			for j in range(n - 1, 0, -1):
				q[j] = q[j] - v[j] * q[j + 1] - w[j] * q[j + 2]

		quincunx(u, v, w, Q, n)
		# Восстанавливаем параметры сплайна
		S[0].d = S[0].y - mu * r[0] * Q[1] * sigma[0]
		if n > 0:
			S[1].d = S[1].y - mu * (f[1] * Q[1] + r[1] * Q[2]) * sigma[1]
			S[0].a = Q[1] / (3.0 * h[0])
			S[0].b = 0.0
			S[0].c = (S[1].d - S[0].d) / h[0] - Q[1] * h[0] / 3.0
		r[0] = 0.0
		for j in range(1, n):
			S[j].a = (Q[j + 1] - Q[j]) / (3.0 * h[j])
			S[j].b = Q[j]
			S[j].c = (Q[j] + Q[j - 1]) * h[j - 1] + S[j - 1].c
			temp = (r[j - 1] * Q[j - 1] + f[j] * Q[j] + r[j] * Q[j + 1])
			S[j].d = S[j].y - mu * temp * sigma[j]
		S[-1].d = S[-1].y
		S[-1].a = 0.0
		S[-1].b = 0.0
		S[-1].c = 0.0
		self.segments = S

	def predict(self, x_new):
		"""
		Предсказание значений сплайна в точках x_new
		"""
		x_new = np.asarray(x_new)
		y_new = np.zeros_like(x_new, dtype=float)
		x = self.x
		S = self.segments
		for i, xq in enumerate(x_new):
			# Находим сегмент
			if xq <= x[0]:
				seg = 0
			elif xq >= x[-1]:
				seg = len(S) - 2
			else:
				seg = np.searchsorted(x, xq) - 1
			s = S[seg]
			dx = xq - s.x
			y_new[i] = s.d + dx * (s.c + dx * (s.b + dx * s.a))
		return y_new

	def get_basis_functions(self):
		"""
		Возвращает список функций-кубиков для каждого сегмента (a, b, c, d)
		"""
		basis = []
		for s in self.segments:
			def f(x, s=s):
				dx = x - s.x
				return s.d + dx * (s.c + dx * (s.b + dx * s.a))

			basis.append(f)
		return basis

	@staticmethod
	def demo():
		"""Демонстрация: визуализация сглаживающего кубического сплайна с разными λ"""

		def plot_tangents(ax, model, x_range, color='orange', length=0.8, linewidth=2):
			x0, xn = x_range[0], x_range[-1]
			y0 = model.predict([x0])[0]
			y1 = model.predict([xn])[0]

			# Приближённые производные через центральную разность
			h = 1e-4
			dy0 = (model.predict([x0 + h])[0] - model.predict([x0 - h])[0]) / (2 * h)
			dy1 = (model.predict([xn + h])[0] - model.predict([xn - h])[0]) / (2 * h)

			xt0 = np.linspace(x0 - length / 2, x0 + length / 2, 50)
			yt0 = y0 + dy0 * (xt0 - x0)
			ax.plot(xt0, yt0, color=color, linestyle='--', label='Касательная слева')

			xt1 = np.linspace(xn - length / 2, xn + length / 2, 50)
			yt1 = y1 + dy1 * (xt1 - xn)
			ax.plot(xt1, yt1, color=color, linestyle=':', label='Касательная справа')

		np.random.seed(42)
		x = np.linspace(0, 2 * np.pi, 20)
		y = np.sin(x) + 0.2 * np.random.randn(len(x))
		xs = np.linspace(x[0], x[-1], 500)
		true_func = np.sin(xs)
		lambdas = [0.1, 0.5, 1.0]

		fig, axes = plt.subplots(1, len(lambdas), figsize=(15, 4), sharex=True, sharey=True)

		for i, lam in enumerate(lambdas):
			spline = CubicClosedSpline()
			spline.fit(x, y, lam=lam)
			ys = spline.predict(xs)

			ax = axes[i] if len(lambdas) > 1 else axes
			ax.plot(x, y, 'ro', label='Данные')
			ax.plot(xs, ys, 'b-', label='Сплайн')
			ax.plot(xs, true_func, 'k--', label='sin(x)')
			plot_tangents(ax, spline, x)

			ax.set_title(f'λ={lam}')
			ax.grid(True)
			if i == 0:
				ax.legend()

		plt.tight_layout()
		plt.show()

	@staticmethod
	def plot(x, y, lam=0.5,
			 show_data=True, color='blue', title=None,
			 num_points=300, figsize=(10, 6), grid=True, legend=True):
		"""
		Визуализация CubicClosedSpline по заданным параметрам.

		Параметры:
		- x, y: входные данные
		- lam: параметр сглаживания λ
		- show_data: показывать ли точки
		- color: цвет линии сплайна
		- title: заголовок
		- num_points: число точек на графике
		- figsize: размер фигуры
		- grid, legend: отображение сетки и легенды
		"""
		x = np.asarray(x)
		y = np.asarray(y)

		spline = CubicClosedSpline()
		spline.fit(x, y, lam=lam)

		x_dense = np.linspace(min(x), max(x), num_points)
		y_pred = spline.predict(x_dense)

		plt.figure(figsize=figsize)
		if show_data:
			plt.scatter(x, y, color='red', alpha=0.6, s=20, label='Данные')
		plt.plot(x_dense, y_pred, color=color, lw=2, label=f'CubicClosedSpline (λ={lam})')

		plt.xlabel("x")
		plt.ylabel("y")
		plt.title(title or "Cubic Closed Spline")
		if grid:
			plt.grid(True)
		if legend:
			plt.legend()
		plt.tight_layout()
		plt.show()


class CardinalZSpline(Spline):
	"""
	Кардинальный Z-сплайн с аналитическим построением через sympy.
	"""

	def __init__(self, m: int):
		"""
		m: порядок сплайна (например, 3 или 4)
		f: функция, значения которой берутся в узлах [-m, ..., m-1]
		"""
		self.m = m
		self.A = self.__finite_difference_matrix(m)

	def fit(self, x: np.ndarray, y: np.ndarray) -> None:
		"""Обучение сплайна на данных"""
		if len(x) != len(y):
			raise ValueError("Sizes for x and y values should be equal")

		current_x = -self.m
		filtered_y = []
		for i in range(len(x)):
			if x[i] == current_x:
				filtered_y.append(y[i])
				current_x += 1

			if current_x == self.m:
				break

		if len(filtered_y) != 2 * self.m:
			raise ValueError(
				f"Wrong number of data points for Z spline: {len(filtered_y)} "
				f"for m value {self.m}. Use range [-m, m - 1] for as x values"
			)

		self.y_m = np.array(filtered_y)
		self.base_function = sym.lambdify(sym.Symbol('x'), self.Z_m())

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""Предсказание значений сплайна в точках x"""
		y_out = []
		for x_i in x:
			y_val = 0
			for j in range(-self.m, self.m):
				y_val += self.y_m[self.m + j] * self.base_function(x_i - j)
			y_out.append(y_val)

		return np.array(y_out)

	def get_basis_functions(self) -> List[Callable[[float], float]]:
		"""Получение базисных функций сплайна"""
		return [self.base_function]

	def __finite_difference_matrix(self, m: int) -> List[List[float]]:
		size = 2 * m - 1
		V = [[(-(m - 1) + i) ** j for j in range(size)] for i in range(size)]
		D = [[0.0] * size for _ in range(size)]
		for i in range(size):
			D[i][i] = 1 / factorial(i)

		V_inv = sym.Matrix(V).inv()
		D_inv = sym.Matrix(D).inv()

		A = D_inv @ V_inv
		return A.tolist()

	def __b1(self, nu: int) -> float:
		coef = 1.0
		for i in range(nu):
			coef *= (-self.m - i)
		return coef / factorial(nu)

	def __b0(self, nu: int) -> float:
		coef = 1.0
		for i in range(nu):
			coef *= (self.m + i)
		return coef / factorial(nu)

	def __B0(self, x: sym.Symbol, p: int, j: int):
		S = sum(self.__b0(nu) * (x - j) ** nu for nu in range(self.m - p))
		return ((x - j) ** p) * S * ((j + 1 - x) ** self.m) / factorial(p)

	def __B1(self, x: sym.Symbol, p: int, j: int):
		S = sum(self.__b1(nu) * (x - j - 1) ** nu for nu in range(self.m - p))
		return ((x - j - 1) ** p) * S * ((x - j) ** self.m) / factorial(p)

	def __Z_p(self, p: int, j: int) -> float:
		sec_id = self.m - j - 1
		if sec_id < 0 or sec_id >= 2 * self.m - 1:
			return 0.0
		return self.A[p][sec_id]

	def Z_m(self, shift: int = 0) -> Piecewise:
		"""
		Выдаёт символьное представление базисной функции Z_m
		"""
		x = sym.Symbol('x')
		pieces = [(0, (x - shift < -self.m) | (x - shift > self.m))]
		for j in range(-self.m, self.m):
			z = sum(
				self.__Z_p(p, j) * self.__B0(x - shift, p, j) +
				self.__Z_p(p, j + 1) * self.__B1(x - shift, p, j)
				for p in range(self.m)
			)
			pieces.append((z, (x - shift >= j) & (x - shift <= j + 1)))
		return Piecewise(*pieces)

	@staticmethod
	def demo(m_values: List[int] = [1, 2, 3, 4, 5, 6], num_points: int = 1000):
		"""
		Строит графики Zₘ(x) при разных m, показывая сходимость к sinc(x)
		"""
		x_vals = np.linspace(-max(m_values), max(m_values), num_points)

		plt.figure(figsize=(10, 6))

		for m in m_values:
			x_nodes = np.arange(-m, m)
			y_nodes = np.zeros_like(x_nodes, dtype=float)

			spline = CardinalZSpline(m=m)
			spline.fit(x_nodes, y_nodes)

			basis_function = spline.get_basis_functions()[0]
			y_base = [basis_function(x) for x in x_vals]

			plt.plot(x_vals, y_base, label=f"Zₘ(x), m={m}")

		sinc_vals = np.sinc(x_vals)
		plt.plot(x_vals, sinc_vals, 'k--', label='sinc(x)', linewidth=1.2)

		plt.title("Сходимость Zₘ(x) к sinc(x)")
		plt.xlabel("x")
		plt.ylabel("Zₘ(x)")
		plt.grid(True)
		plt.axhline(0, color='black', linewidth=0.5)
		plt.legend()
		plt.tight_layout()
		plt.show()

	@staticmethod
	def plot(m, func=None,
			 num_points=1000, figsize=(10, 6),
			 title=None, show_original=True,
			 grid=True, legend=True):
		"""
		Визуализация кардинального Z-сплайна порядка m.

		Параметры:
		- m: порядок сплайна (int)
		- func: функция f(x), которую интерполирует Z-сплайн (если None — импульс)
		- num_points: точек для построения
		- figsize: размер графика
		- title: заголовок
		- show_original: рисовать ли исходную функцию
		- grid, legend: отображать ли сетку и легенду
		"""
		x_nodes = np.arange(-m, m)

		if func is None:
			y_nodes = np.zeros_like(x_nodes, dtype=float)
			y_nodes[m] = 1.0  # дельта-импульс
		else:
			y_nodes = func(x_nodes)

		spline = CardinalZSpline(m=m)
		spline.fit(x_nodes, y_nodes)

		basis_function = spline.get_basis_functions()[0]
		x_vals = np.linspace(-m - 1, m + 1, num_points)
		y_vals = [basis_function(xi) for xi in x_vals]

		plt.figure(figsize=figsize)
		plt.plot(x_vals, y_vals, label=f'Zₘ(x), m={m}', color='blue')

		if func is not None and show_original:
			y_true = func(x_vals)
			plt.plot(x_vals, y_true, 'k--', label='Оригинальная функция')

		plt.title(title or f'Кардинальный Z-сплайн при m={m}')
		plt.xlabel("x")
		plt.ylabel("Zₘ(x)")
		if grid:
			plt.grid(True)
		if legend:
			plt.legend()
		plt.tight_layout()
		plt.show()


# ===== P-Spline расширенный =====

class spline:
	def __init__(self, knots, degree, coefficients=None, dimension=1):
		"""
		Инициализация базового класса Spline.

		Параметры:
		- knots (array-like): Узлы сплайна.
		- degree (int): Степень сплайна.
		- coefficients (array-like, optional): Коэффициенты сплайна.
		- dimension (int): Размерность сплайна.
		"""
		self.knots = np.array(knots)
		self.degree = degree
		self.dimension = dimension
		self.coefficients = np.array(coefficients) if coefficients is not None else None

	def evaluate(self, x):
		"""
		Метод для вычисления значения сплайна в точке x.
		Должен быть переопределен в подклассе.
		"""
		raise NotImplementedError("Этот метод должен быть переопределен в подклассе.")

	def plot_spline(self, x_range, num_points=100):
		"""
		Построение графика сплайна в указанном диапазоне.

		Параметры:
		- x_range (tuple): Кортеж (min_x, max_x) для диапазона построения.
		- num_points (int): Количество точек для построения графика.
		"""
		x_vals = np.linspace(x_range[0], x_range[1], num_points)
		y_vals = self.evaluate(x_vals)

		plt.figure(figsize=(10, 6))
		plt.plot(x_vals, y_vals, label=f"{self.__class__.__name__} сплайн")
		if self.coefficients is not None:
			# Отображаем узлы сплайна
			plt.scatter(self.knots, self.evaluate(self.knots), color='red', label="Узлы сплайна")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title(f"Построение {self.__class__.__name__} сплайна")
		plt.legend()
		plt.grid(True)
		plt.show()

	@classmethod
	def create_p_spline(cls, x, y, knots=None, degree=3, penalty_order=2, lambda_=1.0, dimension=1):
		"""
		Фабричный метод для создания объекта p_spline.

		Параметры:
		- x (array-like): Данные независимой переменной.
		- y (array-like): Данные зависимой переменной.
		- knots (array-like, optional): Узлы сплайна. Если не заданы, будут рассчитаны автоматически.
		- degree (int): Степень сплайна.
		- penalty_order (int): Порядок разностного штрафа.
		- lambda_ (float): Параметр сглаживания.
		- dimension (int): Размерность сплайна.

		Возвращает:
		- p_spline: Экземпляр подкласса p_spline.
		"""
		return p_spline(x, y, knots, degree, penalty_order, lambda_, dimension)


# расширенный p_spline
class p_spline(spline):
	def __init__(self, x, y, knots=None, degree=3, penalty_order=2, lambda_=1.0, dimension=1):
		"""
		Инициализация объекта p_spline.

		Параметры:
		- x (array-like): Данные независимой переменной.
		- y (array-like): Данные зависимой переменной.
		- knots (array-like, optional): Узлы сплайна.
		- degree (int): Степень сплайна.
		- penalty_order (int): Порядок разностного штрафа.
		- lambda_ (float): Параметр сглаживания.
		- dimension (int): Размерность сплайна.
		"""
		self.x = np.array(x)
		self.y = np.array(y)
		self.penalty_order = penalty_order
		self.lambda_ = lambda_
		self.spline = None

		self.knots = knots
		self.degree = degree

		# Инициализируем граничные условия как None
		self.boundary_conditions = None

		# Выполняем подгонку сплайна к данным
		self.fit()

		# Инициализируем базовый класс с найденными параметрами
		super().__init__(knots=self.knots, degree=self.degree, coefficients=self.coefficients, dimension=dimension)

	def _difference_matrix(self, n_bases, d):
		"""
		Создает разностную матрицу порядка d.

		Параметры:
		- n_bases (int): Количество базисных функций B-сплайна.
		- d (int): Порядок разности.

		Возвращает:
		- ndarray: Разностная матрица.
		"""
		D = np.eye(n_bases)
		for _ in range(d):
			D = np.diff(D, n=1, axis=0)
		return D

	def set_boundary_conditions(self, bc_type, bc_values=None):
		"""
		Задает граничные условия для сплайна.

		Параметры:
		- bc_type (str): Тип граничных условий ('natural', 'clamped').
		- bc_values (dict, optional): Значения производных для граничных условий.
			Для 'clamped' требуется {'left': value, 'right': value}.
			Для 'natural' не нужны дополнительные значения.
		"""
		if bc_type not in [None, 'natural', 'clamped', 'cyclic']:
			raise ValueError("Поддерживаемые типы граничных условий: 'natural', 'clamped', 'cyclic'.")

		if bc_type == 'clamped':
			if (bc_values is None or
					'left' not in bc_values or
					'right' not in bc_values):
				raise ValueError(
					"Для 'clamped' граничных условий необходимо предоставить 'left' и 'right' значения производных.")

		self.boundary_conditions = {
			'type': bc_type,
			'values': bc_values
		}

		# После установки граничных условий необходимо повторно выполнить подгонку
		self.fit()

	def fit(self, penalty_fun=None):
		"""
		Аппроксимирует P-сплайн к данным с учетом функции штрафа.

		Параметры:
		- penalty_fun (callable, optional): Функтор для модификации разностной матрицы (например, sin, cos).
		"""
		if self.knots is None:
			num_internal_knots = max(int(len(self.x) / 4), 4)
			self.knots = np.linspace(min(self.x), max(self.x), num_internal_knots)
			self.knots = np.concatenate((
				[self.x[0]] * self.degree,
				self.knots,
				[self.x[-1]] * self.degree
			))

		n = len(self.x)
		t = self.knots
		k = self.degree
		n_bases = len(t) - k - 1

		# Создаем базисную матрицу B
		B = np.zeros((n, n_bases))
		for i in range(n_bases):
			c = np.zeros(n_bases)
			c[i] = 1
			spline = SciPyBSpline(t, c, k)
			B[:, i] = spline(self.x)

		# Создаем разностную матрицу
		D = self._difference_matrix(n_bases, self.penalty_order)

		# Применяем функтор к разностной матрице, если он задан
		if penalty_fun is not None:
			D = penalty_fun(D)
			# Преобразуем элементы в положительные значения
			D = np.abs(D)

		# Создаем штрафную матрицу P
		P = self.lambda_ * D.T @ D

		# Основная система уравнений: (B^T B + P) c = B^T y
		BtB = B.T @ B
		Bty = B.T @ self.y
		A = BtB + P
		rhs = Bty.copy()

		# Сохраняем систему как атрибуты объекта
		self.A = A
		self.rhs = rhs

		# Обработка граничных условий
		if self.boundary_conditions is not None:
			bc_type = self.boundary_conditions['type']
			if bc_type == 'natural':
				# Вторая производная на концах равна нулю
				# Вычисляем вторые производные базисных функций на концах
				# Для каждого базисного сплайна вычисляем его вторую производную на границе
				B_der2_left = np.array([
					SciPyBSpline(t, np.eye(n_bases)[i], k).derivative(2)(self.x[0])
					for i in range(n_bases)
				])
				B_der2_right = np.array([
					SciPyBSpline(t, np.eye(n_bases)[i], k).derivative(2)(self.x[-1])
					for i in range(n_bases)
				])

				# Добавляем эти условия в систему
				self.A = np.vstack([A, B_der2_left, B_der2_right])
				self.rhs = np.hstack([rhs, 0, 0])

			elif bc_type == 'clamped':
				# Первая производная на концах задана
				bc_values = self.boundary_conditions['values']
				# Вычисляем первые производные базисных функций на концах
				B_der1_left = np.array([
					SciPyBSpline(t, np.eye(n_bases)[i], k).derivative(1)(self.x[0])
					for i in range(n_bases)
				])
				B_der1_right = np.array([
					SciPyBSpline(t, np.eye(n_bases)[i], k).derivative(1)(self.x[-1])
					for i in range(n_bases)
				])

				# Добавляем эти условия в систему
				self.A = np.vstack([A, B_der1_left, B_der1_right])
				self.rhs = np.hstack([rhs, bc_values['left'], bc_values['right']])

			elif bc_type == 'cyclic':
				self.set_cyclic_boundary_conditions()

		# Решаем систему уравнений с учетом граничных условий
		try:
			c = np.linalg.lstsq(self.A, self.rhs, rcond=None)[0]
		except np.linalg.LinAlgError as e:
			raise np.linalg.LinAlgError(f"Ошибка при решении системы уравнений: {e}")

		self.coefficients = c
		self.spline = SciPyBSpline(self.knots, c, self.degree)

	def set_cyclic_boundary_conditions(self):
		"""
	  Задает циклические граничные условия для сплайна.
	  """
		if self.knots is None or self.coefficients is None:
			raise ValueError("Сплайн не был инициализирован корректно.")

		n_bases = len(self.coefficients)
		t = self.knots
		k = self.degree

		# Условия на совпадение значений сплайна на концах
		B_start = np.array([
			SciPyBSpline(t, np.eye(n_bases)[i], k)(self.x[0])
			for i in range(n_bases)
		])
		B_end = np.array([
			SciPyBSpline(t, np.eye(n_bases)[i], k)(self.x[-1])
			for i in range(n_bases)
		])
		continuity_row = B_start - B_end

		# Условия на совпадение первой производной
		B_der1_start = np.array([
			SciPyBSpline(t, np.eye(n_bases)[i], k).derivative(1)(self.x[0])
			for i in range(n_bases)
		])
		B_der1_end = np.array([
			SciPyBSpline(t, np.eye(n_bases)[i], k).derivative(1)(self.x[-1])
			for i in range(n_bases)
		])
		derivative1_row = B_der1_start - B_der1_end

		# Нормализация строковых условий
		continuity_row /= np.linalg.norm(continuity_row)
		derivative1_row /= np.linalg.norm(derivative1_row)

		# Увеличиваем вес условий цикличности
		weight = 1e3
		continuity_row *= weight
		derivative1_row *= weight

		# Логирование
		# print("Continuity row:", continuity_row)
		# print("Derivative1 row:", derivative1_row)

		# Обновляем матрицу A и правую часть rhs
		self.A = np.vstack([self.A, continuity_row, derivative1_row])
		self.rhs = np.hstack([self.rhs, 0, 0])

	def evaluate(self, x):
		"""
		Метод для вычисления значения сплайна в точке x.
		"""
		if self.spline is None:
			raise ValueError("Сплайн еще не аппроксимирован.")
		return self.spline(x)

	def predict(self, x_new):
		"""
		Предсказывает значения y для новых значений x с использованием аппроксимированного сплайна.

		Параметры:
		- x_new (array-like): Новые значения x.

		Возвращает:
		- ndarray: Предсказанные значения y.
		"""
		return self.evaluate(x_new)

	def plot_spline(self, x_range=None, num_points=100):
		"""
		Переопределение метода для построения графика сплайна вместе с исходными данными.

		Параметры:
		- x_range (tuple, optional): Диапазон (min_x, max_x) для построения графика. Если не задан, используется диапазон данных.
		- num_points (int): Количество точек для построения графика.
		"""
		if x_range is None:
			x_range = (min(self.x), max(self.x))
		x_vals = np.linspace(x_range[0], x_range[1], num_points)
		y_vals = self.evaluate(x_vals)

		plt.figure(figsize=(10, 6))
		plt.plot(x_vals, y_vals, label=f"{self.__class__.__name__} сплайн")
		plt.scatter(self.x, self.y, color='red', label="Данные")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title(f"Построение {self.__class__.__name__} сплайна")
		plt.legend()
		plt.grid(True)
		plt.show()

	def method_specific_to_p_spline(self):
		"""
		Пример метода, специфичного для p_spline.
		"""

	# print("Это метод, специфичный для p_spline.")

	@staticmethod
	def plot_p_spline(
			start=0, stop=10, num=100, boundary_conditions=None, clamped_values=None,
			penalty_fun=None, point_gen_func="random", power_exp=2, noise_variance=0.0
	):
		"""
		Построение P-сплайна с выбором метода генерации точек и добавлением шума.

		Параметры:
		- start, stop (float): Диапазон значений x.
		- num (int): Количество точек.
		- boundary_conditions (str): Тип граничных условий ('natural', 'clamped').
		- clamped_values (dict): Значения производных для 'clamped' условий.
		- penalty_fun (callable): Функтор для модификации разностной матрицы.
		- point_gen_func (str): Метод генерации точек ('random', 'sin', 'cos', 'exp', 'power').
		- power_exp (float): Экспонента для метода 'power'.
		- noise_variance (float): Дисперсия шума (0.0 = без шума).
		"""
		np.random.seed(None)  # Для случайных точек

		# Генерация точек x и y в зависимости от выбранного метода
		if point_gen_func == "random":
			x_data = np.sort(np.random.uniform(low=start, high=stop, size=num))
			y_data = np.random.uniform(low=-1.0, high=1.0, size=num)  # случайные значения y
		elif point_gen_func == "sin":
			x_data = np.sort(np.random.uniform(low=start, high=stop, size=num))  # Неравноудаленные точки
			y_data = np.sin(x_data)
		elif point_gen_func == "cos":
			x_data = np.sort(np.random.uniform(low=start, high=stop, size=num))  # Неравноудаленные точки
			y_data = np.cos(x_data)
		elif point_gen_func == "exp":
			x_data = np.sort(np.random.uniform(low=start, high=stop, size=num))  # Неравноудаленные точки
			y_data = np.exp(x_data / stop)
		elif point_gen_func == "power":
			x_data = np.sort(np.random.uniform(low=start, high=stop, size=num))  # Неравноудаленные точки
			y_data = x_data ** power_exp
			print(x_data)
		else:
			raise ValueError("Неподдерживаемый метод генерации точек: " + str(point_gen_func))

		if boundary_conditions == "cyclic":
			x_data = np.sort(np.concatenate([x_data, np.array([start, stop])]))
			y_data = np.concatenate([y_data, [y_data[0], y_data[-1]]])  # Добавляем значения на концах

		# Добавляем шум к данным, если задана доля шума
		if noise_variance > 0.0:
			noise_variance = noise_variance / 100  # Преобразуем из процентов в долю
			y_norm = np.sqrt(np.sum(y_data ** 2))  # ||y||_2
			noise_stddev = noise_variance * y_norm  # Масштабируем шум по L2-норме
			noise = np.random.normal(loc=0.0, scale=noise_stddev, size=len(y_data))  # Размер совпадает с y_data
			y_data += noise

		# Создание объекта p_spline
		spline_p = spline.create_p_spline(
			x=x_data,
			y=y_data,
			degree=3,
			penalty_order=2,
			lambda_=1.0
		)
		# Выполняем подгонку с функцией штрафа
		spline_p.fit(penalty_fun=penalty_fun)

		# Построение графика с учетом граничных условий
		print(f"Сплайн с граничными условиями {boundary_conditions}:")
		spline_p.set_boundary_conditions(bc_type=boundary_conditions, bc_values=clamped_values)
		spline_p.plot_spline(x_range=(start, stop), num_points=200)

		if boundary_conditions == 'cyclic':
			# Вывод значений сплайна и его производной на концах
			S_start = spline_p.evaluate(x_data[0])
			S_end = spline_p.evaluate(x_data[-1])
			S_prime_start = spline_p.spline.derivative(1)(x_data[0])
			S_prime_end = spline_p.spline.derivative(1)(x_data[-1])

			print(f"S(x_start) = {S_start}, S(x_end) = {S_end}")
			print(f"S'(x_start) = {S_prime_start}, S'(x_end) = {S_prime_end}")

		# Использование специфичного метода p_spline
		spline_p.method_specific_to_p_spline()


# b_spline
class b_spline(spline):
	def __init__(self, degree, control_points):
		self.control_points = control_points
		self.degree = degree
		super().__init__([], degree)
		self.knots = self.generate_knots()  # Генерация узлового вектора

	def generate_knots(self):
		"""
		Автоматическая генерация узлового вектора.
		"""
		n = len(self.control_points)  # Количество контрольных точек
		m = n + self.degree + 1  # Количество узлов
		knots = [0] * (self.degree + 1)  # Начальные узлы

		# Промежуточные узлы распределены
		interior_knots = np.linspace(1, n - self.degree - 3, m - 2 * (self.degree + 1))  # degree - 1
		# interior_knots = np.linspace(0, n - self.degree, m - 2 * (self.degree + 1))
		knots.extend(interior_knots)
		knots.extend([n - self.degree - 1] * (self.degree + 1))  # Конечные узлы
		return np.array(knots)

	def basis_function(self, i, k, t):
		if k == 0:
			return 1.0 if self.knots[i] <= t < self.knots[i + 1] else 0.0
		else:
			coeff1 = 0.0
			if self.knots[i + k] != self.knots[i]:
				coeff1 = (t - self.knots[i]) / (self.knots[i + k] - self.knots[i]) * self.basis_function(i, k - 1, t)
			coeff2 = 0.0
			if self.knots[i + k + 1] != self.knots[i + 1]:
				coeff2 = (self.knots[i + k + 1] - t) / (
						self.knots[i + k + 1] - self.knots[i + 1]) * self.basis_function(i + 1, k - 1, t)
			return coeff1 + coeff2

	def evaluate(self, t):
		n = len(self.control_points) - 1
		result = np.zeros((len(self.control_points[0]),))

		for i in range(n + 1):
			b = self.basis_function(i, self.degree, t)
			result += b * np.array(self.control_points[i])

		return result

	def plot(self):
		t_values = np.linspace(self.knots[self.degree], self.knots[-self.degree - 1], 100)
		# t_values = np.linspace(self.knots[degree], self.knots[-degree - 1], 100)
		spline_points = np.array([self.evaluate(t) for t in t_values])
		spline_points[-1] = spline_points[-2]

		plt.figure(figsize=(8, 6))
		plt.plot(spline_points[:, 0], spline_points[:, 1], label='B-Сплайн', color='blue')

		control_points = np.array(self.control_points)
		plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label='Контрольные точки')

		plt.title("B-Сплайн")
		plt.xlabel("Ось X")
		plt.ylabel("Ось Y")
		plt.legend()
		plt.grid()
		plt.axis("equal")
		plt.show()

	# Генерация случайных контрольных точек
	def generate_random_control_points(n, x_range=(0, 10), y_range=(0, 10)):
		"""
		Генерирует n случайных контрольных точек.
		"""
		x_coords = np.sort(np.random.uniform(x_range[0], x_range[1], n))
		y_coords = np.random.uniform(y_range[0], y_range[1], n)
		return list(zip(x_coords, y_coords))

	@staticmethod
	def plot_b_spline(degree=2, num=2):
		control_points = b_spline.generate_random_control_points(num)
		spline = b_spline(degree, control_points)
		spline.plot()

# ===== MARS =====

class mars_spline(Spline):

	class basis_function:
		def __init__(self, func):
			self.func = func
			#self.symbolic = None

		def __call__(self, *args):
			return self.func(*args)

		"""Handles f * g"""
		def __mul__(self, other):
			if isinstance(other, mars_spline.basis_function):
				return mars_spline.basis_function(lambda *args: self(*args) * other(*args))
			elif isinstance(other, (int, float)):
				return mars_spline.basis_function(lambda *args: other * self(*args))
			else:
				return NotImplemented

		"""Handles 2 * f"""
		def __rmul__(self, other):
			return self.__mul__(other)

		def __add__(self, other):
			if isinstance(other, mars_spline.basis_function):
				return mars_spline.basis_function(lambda *args: self(*args) + other(*args))
			elif isinstance(other, (int, float)):
				return mars_spline.basis_function(lambda *args: self(*args) + other)
			return NotImplemented

		def __radd__(self, other):
			# This lets `sum()` start with 0
			if isinstance(other, (int, float)):
				return mars_spline.basis_function(lambda *args: other + self(*args))
			return NotImplemented
			#return self.__add__(other)


	"""Multivariate Adaptive Regression Spline"""
	def __init__(self, M_max, x, y, with_pruning=True, d=3, lof='gcv'):
		"""
		Инициализация MARS-сплайна

		Args:
			M_max: Максимальное число базисных функций
			x: Матрица входных данных
			y: Вектор зависимых переменных
			with_pruning: Использовать ли алгоритм backward pass
			d: Гиперпараметр, цена каждой базисной функции. Используется при расчете GCV.
			lof: Lack-of-fit. Можно выбрать 'rss' или 'gcv'
		"""
		x = np.asarray(x)
		if x.ndim == 1:
			n_predictors = 1  # treat 1D array as a single "column"
		elif x.ndim >= 2:
			n_predictors = x.shape[1]
		else:
			raise ValueError("Input does not have dimensions!")  # scalar or empty case
		
		#M_max can't be greater than number of observations 
		#(in that case we would just get same cuts) 
		#(but I'm not sure how it works on higher dimensions)
		M_max |= 1  # sets the least significant bit to 1, making it odd
		self.M_max = M_max#min(M_max, len(y)) 
		self.coefficients = None
		self.basis_functions = np.array([mars_spline.basis_function(lambda x: 1) for _ in range(self.M_max)], dtype=object)
		self.__predictor_indices = list(range(n_predictors))
		self.__used_predictors = np.array([set() for _ in range(self.M_max)], dtype=object)
		self.__with_pruning = with_pruning
		self.d = d
		if lof == 'gcv':
			self.LOF = self.LOF_GCV
		elif lof == 'rss':
			self.LOF = self.LOF_RSS
		else:
			raise ValueError(f"Invalid lof value: {lof}. Must be 'gcv' or 'rss'.")

	"""
	M: Current Basis Function index
	x: Matrix of observations

	returns: B_M, B_M1, v, t, m
	B_M: Candidate basis function (with sign: +1)
	B_M1: Candidate basis function (with sign: -1)
	v: Index of predictor variable used in basis functions B_M and B_M1
	t: Cut point used in basis functions B_M and B_M1
	m: Index of parent basis function that got multiplied with candidate basis functions
	"""
	def candidate_basis_generator(self, M, x):
		for m in range(0, M-1): #начинаем с 0, а не с 1, как в статье, т.к. индексы начинаются с 0
			not_used = [i for i in self.__predictor_indices if i not in self.__used_predictors[m]]
			for v in not_used:
				cut_points = np.unique([x[j, v] for j in range(0, len(x)) if self.basis_functions[m](x[j]) > 0])
				#bf_sum = sum(self.basis_functions[:M-1])

				for t in cut_points:
					B_M = self.basis_functions[m] * self.hinge(v, t, 1)
					B_M1 = self.basis_functions[m] * self.hinge(v, t, -1)
					yield (B_M, B_M1, v, t, m)

		return
	
	def forward_pass(self, x, y):
		M = 2
		while M <= self.M_max:
			lof_star = np.inf
			m_star = None
			v_star = None
			t_star = None
			
			#search for the next best basis functions
			#generator returns new candidate pairs B_M and B_M+1
			for B_M, B_M1, v, t, m in self.candidate_basis_generator(M, x):
				#add candidate basis functions to the model and copmute lof
				basis = np.append(self.basis_functions[:M-1], [B_M, B_M1])
				#compute lack-of-fit of model
				lof = self.LOF(basis, x, y)
				if lof < lof_star:
					lof_star = lof
					m_star = m
					v_star = v
					t_star = t
			
			print(f"added B[{M-1}] = [x{v_star} - {t_star}]_+")
			print(f"added B[{M}] = [-(x{v_star} - {t_star})]_+")
			self.basis_functions[M-1] = self.basis_functions[m_star] * mars_spline.hinge(v_star, t_star, 1)
			self.basis_functions[M] = self.basis_functions[m_star] * mars_spline.hinge(v_star, t_star, -1)
			self.__used_predictors[M-1].add(m_star)
			self.__used_predictors[M].add(m_star)
			M += 2

	def backward_pass(self, x, y):
		M_max = len(self.basis_functions) # number of basis functions
		J_star = set(range(1, M_max + 1)) # {1,2,...,M_max}
		K_star = J_star.copy()
		lof_star = np.inf

		for M in range(M_max, 1, -1): #M_max, M_max-1, M_max-2, ..., 2
			b = np.inf
			L = K_star.copy()
			for m in range(2, M+1):
				K = L.copy()
				K.discard(m) #trying to remove m-th Basis Function

				indices = sorted(i - 1 for i in K)
				remaining_basis = self.basis_functions[indices]
				lof = self.LOF(remaining_basis, x, y)
				if lof <= b:
					b = lof
					K_star = K
				if lof <= lof_star:
					lof_star = lof
					J_star = K
				
		print(f"M_max: {M_max}; J_star: {J_star}; K_star: {K_star}")
		indices = sorted(i - 1 for i in J_star)
		self.basis_functions = self.basis_functions[indices]

	@staticmethod
	def least_squares(basis_funcs, x, y):
		B = np.array([[f(row) for f in basis_funcs] for row in x], dtype=float)
		coefficents, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
		return (B, coefficents)

	def LOF_RSS(self, basis_funcs, x, y):
		B, coeff = mars_spline.least_squares(basis_funcs, x, y)
		y_pred = B @ coeff
		rss = np.sum((y - y_pred) ** 2)
		return rss

	def LOF_GCV(self, basis_funcs, x, y):
		#TODO: Generalized Cross Validation
		## MSE / (1 - Complexity(M)/N)^2
		B, coeff = mars_spline.least_squares(basis_funcs, x, y)
		y_pred = B @ coeff
		mse = np.mean((y - y_pred) ** 2)
		N = len(y)
		C = self.complexity(basis_funcs[1:], x, self.d) #only non-constant basis functions
		return mse / ((1 - C/N) ** 2)

	def complexity(self, basis, x, d = 3):
		#TODO
		#B_ij = (B_i(x_j)), x_j - j-th observation of predictor set x (j-th row in matrix x)
		B = np.array([[f(row) for row in x] for f in basis], dtype=float)
		BTB_inv = np.linalg.pinv(B.T @ B)
		projection = B @ BTB_inv @ B.T
		return np.trace(projection) + 1 + d*len(basis)

	def fit(self, x: np.ndarray, y: np.ndarray) -> None:
		"""Обучение сплайна на данных"""

		#Add basis functions during forward_pass
		self.forward_pass(x, y)

		#Prune model during backward pass
		if self.__with_pruning:
			self.backward_pass(x, y)

		#Compute coefficients
		_, self.coefficients = mars_spline.least_squares(self.basis_functions, x, y)

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""Предсказание значений сплайна в точках x"""
		if self.coefficients is None:
			raise ValueError("Spline not fitted yet")

		x = np.asarray(x)
		if x.ndim == 1:
			x = x.reshape(-1, 1)

		B = np.array([[f(row) for f in self.basis_functions] for row in x], dtype=float)
		return B @ self.coefficients

	def get_basis_functions(self) -> List[Callable]:
		"""Получение базисных функций сплайна"""
		return self.basis_functions

	@staticmethod
	def hinge(index, knot, sign):
		return mars_spline.basis_function(lambda x: max(0, sign * (x[index] - knot)))

	@staticmethod
	def demo(M_max):
		np.random.seed(42)
		x = np.linspace(0, 2 * np.pi, 250)
		x = x.reshape(-1,1)
		x_dense = np.linspace(0, 2 * np.pi, 200)

		# Истинные функции
		y_sin_true = np.sin(5 * x)
		y_sin_true_dense = np.sin(5 * x_dense)
		noise = np.random.normal(0, 0.2, len(x)).reshape(-1,1)
		y_sin_noisy = y_sin_true + noise

		y_poly_true = 0.5 * x ** 2 - x + 1
		y_poly_true_dense = 0.5 * x_dense ** 2 - x_dense + 1
		noise = np.random.normal(0, 0.3, len(x)).reshape(-1,1)
		y_poly_noisy = y_poly_true + noise

		# Параметры
		colors = ['green']

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

		# Аппроксимация синуса
		ax1.scatter(x, y_sin_noisy, alpha=0.6, color='blue', label='Данные с шумом', s=30)
		ax1.plot(x_dense, y_sin_true_dense, 'r--', label='Истинная функция', linewidth=3)

		for color in colors:
			mars_sin = mars_spline(M_max, x, y_sin_noisy)
			mars_sin.fit(x, y_sin_noisy)
			y_sin_pred = mars_sin.predict(x_dense)
			ax1.plot(x_dense, y_sin_pred, color=color, label=f'MARS Sin', linewidth=2)

		ax1.set_title('Аппроксимация синуса', fontsize=14)
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.legend()
		ax1.grid(True, alpha=0.3)

		# Аппроксимация полинома
		ax2.scatter(x, y_poly_noisy, alpha=0.6, color='blue', label='Данные с шумом', s=30)
		ax2.plot(x_dense, y_poly_true_dense, 'r--', label='Истинная функция', linewidth=3)

		for color in colors:
			mars_poly = mars_spline(M_max, x, y_poly_noisy)
			mars_poly.fit(x, y_poly_noisy)
			y_poly_pred = mars_poly.predict(x_dense)
			ax2.plot(x_dense, y_poly_pred, color=color, label=f'MARS Poly', linewidth=2)

		ax2.set_title('Аппроксимация полинома', fontsize=14)
		ax2.set_xlabel('x')
		ax2.set_ylabel('y')
		ax2.legend()
		ax2.grid(True, alpha=0.3)

		plt.tight_layout()
		plt.show()

	@staticmethod
	def plot(x, y, M_max,
			 show_data=True, color='blue', title=None,
			 num_points=300, figsize=(10, 6), grid=True, legend=True,
			 x_start=None, x_end=None, lof='rss'):
		"""
		Визуализация MARS-сплайна с параметрами.
		"""
		spline = mars_spline(M_max, x, y, lof=lof)
		spline.fit(x, y)

		x_start = min(x) if x_start is None else x_start
		x_end = max(x) if x_end is None else x_end
		x_dense = np.linspace(x_start, x_end, num_points)
		y_pred = spline.predict(x_dense)

		spline_without_pruning = mars_spline(M_max, x, y, with_pruning=False,lof=lof)
		spline_without_pruning.fit(x, y)
		y_pred_2 = spline_without_pruning.predict(x_dense)

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
		#plt.figure(figsize=figsize)
		if show_data:
			ax1.scatter(x, y, color='red', alpha=0.6, label='Данные', s=20)
		ax1.plot(x_dense, y_pred, color=color, label=f'MARS M_max={M_max}')
		ax1.set_xlabel("x")
		ax1.set_ylabel("y")
		ax1.set_title(title or f"MARS PRUNED ({len(spline.basis_functions)} basis functions)")
		
		if show_data:
			ax2.scatter(x, y, color='red', alpha=0.6, label='Данные', s=20)
		ax2.plot(x_dense, y_pred_2, color=color, label=f'MARS M_max={M_max}')
		ax2.set_xlabel("x")
		ax2.set_ylabel("y")
		ax2.set_title(title or f"MARS ({len(spline_without_pruning.basis_functions)} basis functions)")

		if grid:
			ax1.grid(True)
			ax2.grid(True)
		if legend:
			ax1.legend()
			ax2.legend()

		plt.tight_layout()
		plt.show()

# Для отладки
if __name__ == "__main__":
	#p_spline.plot_p_spline()

	X = np.array([[-4], [-3], [-2], [-1], [0], [1], [2], [3], [4]])
	y = np.array([[5], [4], [3], [1.6], [1.3], [1.1],[3.1],[6.1],[8.1]])

	mars_spline.demo(40)

	#mars_spline.plot(X, y, M_max=40, x_start=-5, x_end=5, num_points=1000,lof='gcv')
	pass

# Для отладки
if __name__ == "__main__":
	#p_spline.plot_p_spline()
	pass