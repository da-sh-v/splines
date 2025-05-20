import numpy as np


# Универсальная функция генерации точек
def generate_noisy_data(func, noise_param=0.0,
						x_range=(0, 10), n_points=50,
						return_true_curve=True, dense_resolution=500):
	"""
	Генерация зашумлённых данных и (опционально) истинной кривой.

	Параметры:
		func         — математическая функция (например, np.sin)
		noise_param  — амплитуда шума
		x_range      — диапазон по оси X: (min, max)
		n_points     — количество точек
		return_true_curve — возвращать ли "истинную" функцию
		dense_resolution — сколько точек в плотной сетке

	Returns:
		x, y —  данные
		x_dense, y_true — (если return_true_curve=True) плотная сетка и истинная функция
	"""

	# функция шума
	def w_noise(a, N):
		return a * np.random.normal(0, 1, N)

	x = np.sort(np.random.uniform(low=x_range[0], high=x_range[1], size=n_points))
	y = func(x)
	y += w_noise(noise_param, len(x))

	# Плотная сетка и истинная кривая
	if return_true_curve:
		x_dense = np.linspace(x_range[0], x_range[1], dense_resolution)
		y_true = func(x_dense)
		return x, y, x_dense, y_true
	else:
		return x, y