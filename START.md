###  Использование:

1. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

2. Запустите скрипт:

   ```bash
   python main.py
   ```

В проекте используется версия Python 3.12.3. Если возникают ошибки по поводу библиотек, это может быть связано с использованием конфигурации Python из другого проекта. Решение: активировать виртуальное окружение текущего проекта или установить зависимости в нужной среде.

### Структура проекта:

1. **main.py**: Главный файл запуска. Связывает интерфейс (GUI) и библиотеки для построения сплайнов.

   - Путь: `/splines/main.py`

2. **splines.py**: Реализация базового класса Spline и его наследников (B-spline и P-spline).

   - Путь: `/splines/code_interfaces/splines.py`

3. **interface.py**: Логика взаимодействия интерфейса с функциями сплайнов. Подключает GUI к библиотеке сплайнов.

   - Путь: `/splines/code_interfaces/interface.py`

4. **Файлы интерфейса (.ui)**:

   - Путь: `/splines/code_interfaces/ui/`
   
   - `main.ui`: Главный интерфейс с настройками и выбором сплайнов.
   - `b_variable.ui`, `p_variable.ui`: Настройки для различных типов сплайнов.

5. **README.md**: Лицевая страница проекта, краткое описание, цели и контакты руководителя/учатсников .

   - Путь: `/splines/README.md`

6. **START.md**: Этот файл содержит инструкции по работе с проектом и его запуску.

   - Путь: `/splines/START.md`

### Запуск проекта:

#### 1. Режим GUI:

Для построения графиков с помощью интерфейса:

```bash
python main.py
```

Выберите нужный тип сплайна, настройте параметры и нажмите "Построить" для отображения результата.

#### 2. Использование без GUI:

Описание функции для построения P-сплайна(с указанными по умолчанию параметрами):

```python
 def plot_p_spline(
        start=0, 
        stop=10, 
        num=100, 
        boundary_conditions=None, 
        clamped_values=None,
        penalty_fun=None, 
        point_gen_func="random", 
        power_exp=2, 
        noise_variance=0.0
 ):
   #код функции
```
### Параметры функции plot_p_spline:

- `start` (float): Начало диапазона x.(график от - до)
- `stop` (float): Конец диапазона x.
- `num` (int): Количество точек.
- `boundary_conditions` (str): Тип граничных условий (`'natural'`, `'clamped'`, `cyclic`).
- `clamped_values` (dict): Значения производных для `'clamped'` условий.
   Задаются таким образом:
```python
clamped_values = {'left': 1,'right': 1}
```
- `penalty_fun` (callable): Штрафная функция, в параметр передается функтор, например sin.
- `point_gen_func` (str): Метод генерации точек (`'random'`, `'sin'`, `'cos'`, `'exp'`, `'power'`).
- `power_exp` (float): Степень для метода `'power'`(x^a).
- `noise_variance` (float): Дисперсия шума (по умолчанию 0.0).

### Пример вызова:

```python
from code_interfaces.splines import p_spline

# Построение P-сплайна с параметрами по умолчанию

p_spline.plot_p_spline()

# Построение P-сплайна с использованием различных методов генерации данных
p_spline.plot_p_spline(
    start=0,
    stop=10,
    num=100,
    boundary_conditions='natural',
    clamped_values=None,
    penalty_fun=None,
    point_gen_func='random',
    power_exp=2,
    noise_variance=0.1
)
```
