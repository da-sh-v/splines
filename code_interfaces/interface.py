from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QListWidgetItem
from code_interfaces.splines import p_spline, b_spline, PTSpline, SmoothingCubicSpline, CubicClosedSpline, CardinalZSpline
import sys
from numpy import sin, cos, exp

test=None	#глобальная переменная для теста, в данном случае если нажать test настраиваемое окно не закроется после построения
class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()

		# Загружаем интерфейс из файла .ui
		uic.loadUi("code_interfaces/ui/main.ui", self)

		# Подключение кнопок выполнить, тест и выход к соответствующим методам
		self.buttonExecute.clicked.connect(self.on_execute)
		self.pushButton.clicked.connect(self.test_variable_p_spline)
		self.buttonExit.clicked.connect(self.quit_program)

		# Список радиокнопок для типов сплайнов
		self.spline_buttons = [
			self.p_radioButton,
			self.pt_radioButton,
			self.z_radioButton,
			self.sm_radioButton
		]

		# Подключаем каждую радиокнопку к общему обработчику связи вида сплайна и действия
		for button in self.spline_buttons:
			button.toggled.connect(self.update_listWidget)

		# Подключаем событие для отслеживания изменений в QListWidget с действиями
		self.listWidget.itemChanged.connect(self.on_item_checked)

		# Инициализация начальных элементов списка
		self.update_listWidget()
	def quit_program(self):
		sys.exit()
	def update_listWidget(self):
		"""Обновляет элементы QListWidget в зависимости от выбранного типа сплайна."""
		# Очищаем предыдущие элементы
		self.listWidget.clear()

		# Добавляем элементы в зависимости от выбранного сплайна
		items = []
		if self.p_radioButton.isChecked():
			items = ["Опции для p-сплайна"]
		elif self.z_radioButton.isChecked():
			items = ["Опции для z-сплайна"]
		elif self.pt_radioButton.isChecked():
			items = ["Опции для pt-сплайна"]
		elif self.sm_radioButton.isChecked():
			items = ["Опции для сгл_куб-сплайна"]

		# Добавляем элементы в QListWidget с флажками
		for item_text in items:
			item = QListWidgetItem(item_text)
			item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
			item.setCheckState(Qt.CheckState.Unchecked)
			self.listWidget.addItem(item)

	def on_item_checked(self, item):
		"""Обработчик изменения состояния флажка в QListWidget."""
		state = "Выбранный" if item.checkState() == Qt.CheckState.Checked else "Снята отметка"
		print(f"{state} элемент: {item.text()}")

	def on_execute(self):
		"""Проверяет, выбран ли p-сплайн, и выполняет соответствующее действие."""
		global test
		test = None
		if self.p_radioButton.isChecked():
			if self.ExampleButton.isChecked():
				self.handle_p_spline()
			elif self.VariableButton.isChecked():
				self.run_p_spline_slider_window()
			else:
				self.label_output.setText("Выберите пример.")

		elif self.pt_radioButton.isChecked():
			if self.ExampleButton.isChecked():
				self.handle_pt_spline() 				# доделать
			else:
				self.label_output.setText("Выберите пример.")

		elif self.sm_radioButton.isChecked():
			if self.ExampleButton.isChecked():
				self.handle_sm_spline()				# доделать
			elif self.VariableButton.isChecked():
				self.handle_ccs_spline()
			else:
				self.label_output.setText("Выберите пример.")

		elif self.z_radioButton.isChecked():
			if self.ExampleButton.isChecked():
				self.handle_z_spline()				# доделать
			else:
				self.label_output.setText("Выберите пример.")



	def development(self):
		self.label_output.setText("все, кроме p-сплайна, пока в разработке.")

	def test_variable_p_spline(self):
		global test
		test=True
		self.slider_window = SliderWindow_p()
		self.slider_window.show()


	def handle_p_spline(self):
		"""Выводит сообщение и отображает график для p-сплайна."""
		p_spline.plot_p_spline()
		self.label_output.setText("График p-сплайн был выбран и нажата кнопка Выполнить.")

	def handle_z_spline(self):
		CardinalZSpline.demo()
		self.label_output.setText("График z-сплайн был выбран и нажата кнопка Выполнить.")

	def handle_pt_spline(self):
		PTSpline.demo()
		self.label_output.setText("График pt-сплайн был выбран и нажата кнопка Выполнить.")

	def handle_sm_spline(self):
		PTSpline.demo()
		self.label_output.setText("График сгл_куб-сплайн был выбран и нажата кнопка Выполнить.")

	def handle_pt_spline(self):
		PTSpline.demo()
		self.label_output.setText("График сгл_куб-сплайн с закрепленными концами был выбран и нажата кнопка Выполнить.")

	def handle_ccs_spline(self):
		CubicClosedSpline.demo()
		self.label_output.setText("График pt-сплайн был выбран и нажата кнопка Выполнить.")


	#слайдер пока реализован только для p-сплайна и b-сплайна
	def run_p_spline_slider_window(self):
		"""Открывает окно с ползунками для выбора значений."""
		self.label_output.setText("График p-сплайн был выбран и нажата кнопка Выполнить.")
		self.slider_window = SliderWindow_p()
		self.slider_window.show()

	def run_b_spline_slider_window(self):
		"""Открывает окно с ползунками для выбора значений."""
		self.label_output.setText("График b-сплайн был выбран и нажата кнопка Выполнить.")
		self.slider_window = SliderWindow_b()
		self.slider_window.show()


class SliderWindow_p(QWidget):
	def __init__(self):
		super().__init__()
		uic.loadUi('code_interfaces/ui/p_variable.ui', self)
		# Привязка сигналов ползунков и кнопки к методам
		self.slider_start.valueChanged.connect(self.update_start)
		self.slider_stop.valueChanged.connect(self.update_stop)
		self.slider_num.valueChanged.connect(self.update_num)
		self.slider_clamped_values_left.valueChanged.connect(self.update_clamped_left)
		self.slider_clamped_values_right.valueChanged.connect(self.update_clamped_right)
		self.slider_power.valueChanged.connect(self.update_power)
		self.slider_noise.valueChanged.connect(self.update_noise)

		self.button_apply.clicked.connect(self.validate_values)

	def update_start(self, value):
		"""Обновление начальной точки."""
		self.label_start.setText(f"Start: {value}")

	def update_stop(self, value):
		"""Обновление конечной точки."""
		self.label_stop.setText(f"Stop: {value}")

	def update_num(self, value):
		"""Обновление количества точек."""
		self.label_num.setText(f"Количество точек:: {value}")

	def update_clamped_left(self, value):
		self.label_clamped_left.setText(f"Clamped values: {value}")

	def update_clamped_right(self, value):
		self.label_clamped_right.setText(f"Clamped values: {value}")

	def update_power(self, value):
		self.label_power.setText(f"a= {value}")

	def update_noise(self, value):
		self.label_noise.setText(f"Шум: {value}%")

	def validate_values(self):
		"""Проверка и вывод значений ползунков."""
		start = self.slider_start.value()
		stop = self.slider_stop.value()
		num = self.slider_num.value()
		noise = self.slider_noise.value()
		clamped_values = {'left': self.slider_clamped_values_left.value(),
						  'right': self.slider_clamped_values_right.value()}
		power_exp = self.slider_power.value()

		if self.radioButton_bc_natural.isChecked():
			boundary_conditions = 'natural'
		elif self.radioButton_bc_clamped.isChecked():
			boundary_conditions = 'clamped'
		elif self.radioButton_bc_cyclic.isChecked():
			boundary_conditions = 'cyclic'
		else:
			boundary_conditions = None

		if self.radioButton_penalty_fun_sin.isChecked():
			penalty_fun = sin
		elif self.radioButton_penalty_fun_cos.isChecked():
			penalty_fun = cos
		elif self.radioButton_penalty_fun_exp.isChecked():
			penalty_fun = exp
		else:
			penalty_fun = None

		if self.radioButton_generation_sin.isChecked():
			point_gen_func = "sin"
		elif self.radioButton_generation_cos.isChecked():
			point_gen_func = "cos"
		elif self.radioButton_generation_exp.isChecked():
			point_gen_func = "exp"
		elif self.radioButton_generation_power.isChecked():
			point_gen_func = "power"
		else:
			point_gen_func = "random"

		if start >= stop:
			print("Ошибка: Start должен быть меньше Stop!")
		elif num < 2:
			print("Ошибка: количество точек должно быть больше или равно 2!")
		else:
			print(
				f"Start: {start}\n"
				f"Stop: {stop}\n"
				f"Num: {num}\n"
				f"Boundary conditions: {boundary_conditions}\n"
				f"Clamped values по y: {clamped_values}\n"
				f"Penalty fun: {penalty_fun}\n"
				f"Point gen func: {point_gen_func}\n"
				f"Power exp: {power_exp}\n"
				f"Noise: {noise}%"
			)
			p_spline.plot_p_spline(start=start, stop=stop, num=num,
								   boundary_conditions=boundary_conditions, clamped_values=clamped_values,
								   penalty_fun=penalty_fun, point_gen_func=point_gen_func,
								   power_exp=power_exp, noise_variance=noise)
			if test is None:
				self.close()

class SliderWindow_b(QWidget):
	def __init__(self):
		super().__init__()
		uic.loadUi('code_interfaces/ui/b_variable.ui', self)
		# Привязка сигналов ползунков и кнопки к методам
		self.slider_start.valueChanged.connect(self.update_start)
		self.slider_stop.valueChanged.connect(self.update_stop)
		self.slider_num.valueChanged.connect(self.update_num)
		self.slider_degree.valueChanged.connect(self.update_degree)

		self.button_apply.clicked.connect(self.validate_values)

	def update_start(self, value):
		"""Обновление начальной точки."""
		self.label_start.setText(f"Start: {value}")

	def update_stop(self, value):
		"""Обновление конечной точки."""
		self.label_stop.setText(f"Stop: {value}")

	def update_num(self, value):
		"""Обновление количества точек."""
		self.label_num.setText(f"Количество точек: {value}")
	def update_degree(self, value):
		"""Обновление количества точек."""
		self.label_degree.setText(f"Степень: {value}")

	def validate_values(self):
		"""Проверка и вывод значений ползунков."""
		start=self.slider_start.value()
		stop=self.slider_stop.value()
		num=self.slider_num.value()
		degree=self.slider_degree.value()

		if start >= stop:
			print("Ошибка: Start должен быть меньше Stop!")
		elif num < 2:
			print("Ошибка: количество точек должно быть больше или равно 2!")
		else:
			print(f"Degree: {degree}, Num: {num}")
			b_spline.plot_b_spline(degree, num)

def start():
	"""Создает и запускает приложение."""
	app = QApplication.instance()
	if app is None:  # Проверяем, запущено ли приложение
		app = QApplication([])

	window = MainWindow()
	window.show()

	if not app.exec():  # Запускаем цикл событий только один раз
		app.exec()
