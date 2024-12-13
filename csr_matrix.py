import typing as tp

# юнит-тесты напишу к среде (включительно) (ко всем своим методам) - Илья
# все комментарии, сделанные через решетку - для отчета


class csr_matrix:
    def __init__(  # конструктор (будет использоваться в юнит-тестах)
        self,
        n: tp.Optional[int] = None,
        m: tp.Optional[int] = None,
        column_indices: tp.Optional[tp.List[int]] = None,
        row_pointers: tp.Optional[tp.List[int]] = None,
        values: tp.Optional[tp.List[float]] = None,
    ) -> None:
        """
        Основной констуктор матриц разреженного-строчного типа хранения
        (Для тестов)

        :param n: Кол-во строк
        :param m: Кол-во столбцов
        :param values: Массив ненулевых элементов
        :param column_indices: Массив индексов столбцов каждого элемента values
        :param row_pointers: Массив индексов values, указывающих на начало каждой строки
        :return: Экземпляр csr_matrix
        """
        self.n = n
        self.m = m
        self.column_indices = column_indices.copy() if column_indices is not None else []
        self.row_pointers = row_pointers.copy() if row_pointers is not None else []
        self.values = values.copy() if values is not None else []

    def fill_from_input(self) -> None:  # метод для заполнения значений из потока ввода
        """
        Заполнить матрицу разреженного-строчного типа через построчный ввод

        :return: None
        """
        inpt = list(input().split()) # записываем n и m

        if len(inpt) > 2 or len(inpt) == 0:
            raise ValueError("Invalid number of boundaries")
        
        if len(inpt) == 1:
            self.n = int(inpt[0])
            self.m = self.n
        else:
            self.n, self.m = map(int, inpt)

        if self.m <= 0 or self.n <= 0:
            raise ValueError("Invalid matrix boundaries")

        for i in range(self.n):
            temporary_row = list(map(float, input().split()))  # считываем ввод построчно и
            # создаем из строки массив

            if len(temporary_row) != self.m:
                raise ValueError("Invalid length of row")

            self.row_pointers.append(len(self.values))  # фиксируем конец последней строки
            # и начало следующей

            for j in range(self.m):
                if temporary_row[j] == 0:
                    continue
                else:
                    self.values.append(float(temporary_row[j]))  # записываем только ненулевые значения
                    self.column_indices.append(j)

        self.row_pointers.append(len(self.values))  # фиксируем конец последней строки

    def fill_from_matrix(self, matrix: tp.List[tp.List[float]]) -> None:
        """
        Заполнить данную csr_matrix на основе матрицы в обычном представлении

        :param matrix: Матрица, из которой нужно взять значения
        :return: None
        """
        # буквально аналог функции fill_from_input, но только для тестов
        # так как сильно схожа с fill_from_input, тесты не нужны
        self.n, self.m = len(matrix), len(matrix[0])

        for i in range(self.n):

            if len(matrix[i]) != self.m:
                raise AttributeError("Passed a non-matrix")

            self.row_pointers.append(len(self.values))

            for j in range(self.m):
                if matrix[i][j] == 0:
                    continue
                else:
                    self.values.append(float(matrix[i][j]))
                    self.column_indices.append(j)
        self.row_pointers.append(len(self.values))

    def __getitem__(self, coordinates: tp.Tuple[int, int]) -> float:  # метод для получения значения по указанным координатам
        """
        Получить значение матрицы, стоящее на ячейке с координатами coordinates

        :param coordinates: Координаты запрашиваемого значения (формат - пара чисел, Tuple)
        :return: Значение по указанным координатам
        """
        if self.n is None or self.m is None:
            raise AttributeError("Can't find value by coords for empty matrix")  # проверяем матрицу на пустоту

        if (
            len(coordinates) != 2
            or not isinstance(coordinates[0], int)
            or not isinstance(coordinates[1], int)
            or (self.n < coordinates[0] or self.m < coordinates[1])
            or (coordinates[0] <= 0 or coordinates[1] <= 0)
        ):
            raise ValueError("Invalid coordinates")  # если координаты выходят за пределы матрицы или переданы отрицательные

        row_start, row_end = (
            self.row_pointers[coordinates[0] - 1],
            self.row_pointers[coordinates[0]],
        )  # указатели на границы i-ой строки находятся на i-ой и i+1-ой позициях
        # (по способому задания данного массива); так как на вход подаются
        # координаты с нумерацией с началом в 1 -> берем на 1 меньше

        for i in range(row_start, row_end):  # проверяем строку на наличие данного индекса столбца
            if self.column_indices[i] == coordinates[1] - 1:
                return self.values[i]  # если нашли - возвращаем значение
            if self.column_indices[i] > coordinates[1] - 1:
                break  # так как отрезки массива монотонные - после данного индекса в пределах одной строки
                # будут только индексы, большие данного
        return 0.0  # если не нашли нужный индекс столбца - значит, на этом месте стоит 0

    def __setitem__(self, coordinates: tp.Tuple[int, int], value: float):
        """
        Метод для установки значения по указанным координатам
        
        :param coordinates: Координаты для вставки или замены
        :param value: вставляемое или заменяемое значение
        """
        if self.n is None or self.m is None:
            raise AttributeError("Can't find value by coords for empty matrix")  # проверяем матрицу на пустоту
        
        if (
            len(coordinates) != 2
            or not isinstance(coordinates[0], int)
            or not isinstance(coordinates[1], int)
            or (self.n < coordinates[0] or self.m < coordinates[1])
            or (coordinates[0] <= 0 or coordinates[1] <= 0)
        ):
            raise ValueError("Invalid coordinates")  # если координаты выходят за пределы матрицы или переданы отрицательные
        
        row_start, row_end = (
            self.row_pointers[coordinates[0] - 1],
            self.row_pointers[coordinates[0]],
        )
        i = row_start
        for i in range(row_start, row_end):  # проверяем строку на наличие данного индекса столбца
            if self.column_indices[i] == coordinates[1] - 1:
                self.values[
                    i] = value  # если нашли - устанавливаем значение, если ноль, то можно было бы сдвинуть, чтобы сохранить разряженность, но не будем для снижения асимптотической сложности
                return
            if self.column_indices[i] > coordinates[1] - 1:
                break
                #  если не нашли - вставляем значение и сдвигаем указатели
        self.values.insert(i, value)
        self.column_indices.insert(i, coordinates[1] - 1)
        for k in range(coordinates[0], len(self.row_pointers)):
            self.row_pointers[k] += 1

    def get_matrix_trace(self) -> float:  # метод для получения следа матрицы
        """
        Получить след матрицы

        :return: След матрицы
        """
        if self.n is None or self.m is None:
            raise AttributeError("Can't find trace for empty matrix")  # проверяем матрицу на пустоту

        if self.n != self.m:
            raise AttributeError(
                "Matrix must be square to get matrix's trace"
            )  # след матрицы существует только для квадратных; здесь это проверяем

        trace = 0.0
        for i in range(1, self.n + 1):
            trace += self[i, i]  # складываем все элементы на главной диагонали

        return trace

    def __add__(self, other) -> tp.Self:  # перегрузка оператора сложения (для реализации метода сложения матриц)

        if not isinstance(other, csr_matrix):
            raise AttributeError("Can't sum a matrix and a non-matrix type") # проверка на типизацию

        if self.n != other.n or self.m != other.m:
            raise AttributeError("Can't sum matrices of different sizes") # проверка размерностей

        mtrx_sum = csr_matrix(n=self.n, m=self.m) # сюда будем записывать значения суммы матриц
        for i in range(self.n): # проходим по строкам
            row_start_1, row_end_1, row_start_2, row_end_2 = (
                self.row_pointers[i],
                self.row_pointers[i + 1],
                other.row_pointers[i],
                other.row_pointers[i + 1],
            ) # складываем поэлементно и построчно с помощью метода двух указателей

            i_1 = row_start_1 # первый указатель (проходит по строке первой матрицы)
            i_2 = row_start_2 # второй указатель (проходит по строке второй матрицы)

            mtrx_sum.row_pointers.append(len(mtrx_sum.values)) # фиксируем конец предыдущей строки и начало текущей
            
            for j in range(self.m):
                
                # поддерживаем условие, индекс столбца указателя больше или равен текущему
                while i_1 < row_end_1 - 1 and j > self.column_indices[i_1]: 
                    i_1 += 1

                # то же самое
                while i_2 < row_end_2 - 1 and j > other.column_indices[i_2]:
                    i_2 += 1

                sum_ = 0.0

                # если указатель совпал с рассматриваемым столбцом -> складываем
                if i_1 < row_end_1 and self.column_indices[i_1] == j:
                    sum_ += self.values[i_1]

                if i_2 < row_end_2 and other.column_indices[i_2] == j:
                    sum_ += other.values[i_2]

                # записываем
                if sum_ != 0:
                    mtrx_sum.values.append(sum_)
                    mtrx_sum.column_indices.append(j)

        # фиксируем конец последней строки
        mtrx_sum.row_pointers.append(len(mtrx_sum.values))

        return mtrx_sum

    def __mul__(
        self, other
    ) -> tp.Self:  # перегрузка оператора умножения (для реализации метода умножения матриц друг на друга и домножения на скаляр)
        if not isinstance(other, (csr_matrix, float, int)):
            raise AttributeError("Invalid multiplication") # проверка типизации

        if isinstance(other, (float, int)):  # умножение на скаляр
            mtrx_mul = csr_matrix(n=self.n, m=self.m) # создаем новый экземпляр матрицы
            if other == 0.0:
                mtrx_mul.row_pointers = [0] * (self.n + 1) # если скаляр нулевой -> все массивы пустые (кроме указателей строк)
            else:
                # если же скаляр ненулевой -> создаем копии всех массивов изначальной матрицы и домножаем values на скаляр
                mtrx_mul.column_indices = self.column_indices.copy() 
                mtrx_mul.row_pointers = self.row_pointers.copy()
                mtrx_mul.values = self.values.copy()
                for i in range(len(self.values)):
                    mtrx_mul.values[i] *= other

        else:  # умножение матриц
            if self.m != other.n:
                raise AttributeError("Can't multiply matrices of non-compatible sizes") # проверка на размерность

            mtrx_mul = csr_matrix(n=self.n, m=other.m) # создаем новую матрицу, соответствующую размерности произведения

            for i in range(self.n): # заполняем матрицу произведения поэлементно (здесь рассматриваем строку)

                mtrx_mul.row_pointers.append(len(mtrx_mul.values)) # фиксируем границу предыдущей строки и начало текущей

                for j in range(other.m): # здесь рассматриваем столбец матрицы умножения
                    mul_ = 0.0

                    for k in range(self.m):
                        mul_ += self[i + 1, k + 1] * other[k + 1, j + 1] # проходимся по строке левой матрицы и по столбцу
                                                                         # правой -> вычисляем элемент матрицы произведения
                    if mul_ == 0: # запись элемента в матрицу произведения
                        continue
                    else:
                        mtrx_mul.values.append(mul_)
                        mtrx_mul.column_indices.append(j)
            # фиксируем границу последней строки
            mtrx_mul.row_pointers.append(len(mtrx_mul.values))

        return mtrx_mul

    def __rmul__(self, other) -> tp.Self:  # метод обратного умножения для коммутативности умножения на скаляр
        if not isinstance(other, (float, int)):
            raise AttributeError("Invalid multiplication")

        return self.__mul__(other)

    def __str__(self) -> str:  # метод для вывода экземпляра класса в обычном матричном виде (для относительно малых матриц)
        if self.n is None or self.m is None:
            return ""
        matrix = ""

        for i in range(self.n):
            for j in range(self.m):
                matrix += f"{self[i + 1, j + 1]} "
            matrix += "\n"

        return matrix

    def __eq__(self, other): # метод сравнения (равенства) для адекватного assert'а в тестах
        if not isinstance(other, csr_matrix):
            raise AttributeError("Can't compare a matrix and a non-matrix")

        if self.n != other.n or self.m != other.m:
            return False

        if (
            len(self.column_indices) != len(other.column_indices)
            or len(self.values) != len(other.values)
            or len(self.row_pointers) != len(other.row_pointers)
        ):
            return False

        if not all(self.column_indices[i] == other.column_indices[i] for i in range(len(self.column_indices))):
            return False

        if not all(self.row_pointers[i] == other.row_pointers[i] for i in range(len(self.row_pointers))):
            return False

        if not all(float(self.values[i]) == float(other.values[i]) for i in range(len(self.values))):
            return False

        return True

    def swap_lines(self, line_a: int, line_b: int):
        """
        Метод для замены строк матрицы

        :param line_a: индекс первой линии для замены
        :param line_b: индекс второй линии для замены
        """
        if line_a > self.n or line_b > self.n or line_a < 0 or line_b < 0:
            raise AttributeError("Invalid line index")
        
        row1_start, row1_end = (
            self.row_pointers[line_a - 1],
            self.row_pointers[line_a],
        )

        row2_start, row2_end = (
            self.row_pointers[line_b - 1],
            self.row_pointers[line_b],
        )

        if (row1_end - row1_start) == 0 and (row2_end - row2_start) == 0:
            return  # если обе строки пустые, то ничего менять не надо

        section1 = self.values[row1_start:row1_end]
        section2 = self.values[row2_start:row2_end]
        section1_cols = self.column_indices[row1_start:row1_end]
        section2_cols = self.column_indices[row2_start:row2_end]

        # Определяем порядок индексов для корректного удаления и вставки
        if row1_start > row2_start:
            # Меняем порядок, чтобы сначала обработать второй участок
            row1_start, row1_end, row2_start, row2_end = row2_start, row2_end, row1_start, row1_end
            section1, section2 = section2, section1
            section1_cols, section2_cols = section2_cols, section1_cols

        # Удаление старых участков
        del self.values[row2_start:row2_end]
        del self.values[row1_start:row1_end]
        del self.column_indices[row2_start:row2_end]
        del self.column_indices[row1_start:row1_end]

        # Вставка участков в правильном порядке
        for i, value in enumerate(section2):
            self.values.insert(row1_start + i, value)
        for i, value in enumerate(section2_cols):
            self.column_indices.insert(row1_start + i, value)
        insertion_correction = (row1_end - row1_start) - (
                    row2_end - row2_start)  # коррекция row_pointers для вставки второй строки относительно замены одной строку на другую
        for i, value in enumerate(section1):
            self.values.insert(row2_start + i - insertion_correction, value)
        for i, value in enumerate(section1_cols):
            self.column_indices.insert(row2_start + i - insertion_correction, value)

        # коррекция ссылок на строки
        diff = (row1_end - row1_start) - (row2_end - row2_start)
        for i in range(line_a, line_b):
            self.row_pointers[i] -= diff

    def get_matrix_memory(self) -> tp.List[tp.List[float]]:
        """
        Создаём двумерный массив из разрежённой матрицы

        :return: Матрица в виде двумерного массива
        """
        matrix = []
        if self.n is None or self.m is None:
            return matrix

        for i in range(self.n):
            row = []
            for j in range(self.m):
                row.append(self[i + 1, j + 1])
            matrix.append(row)
        return matrix

    def get_determinant(self) -> float:
        """
        метод для округления детерминанта до 4 знаков после запятой
        :return: детерминант, с точностью 4 знаков после запятой
        """
        rounded = round(self.get_raw_determinant(), 4)
        return rounded

    def get_raw_determinant(self) -> float:
        """
        Вычисляет определитель матрицы методом Гаусса.
        :return: Определитель матрицы
        """
        if self.n != self.m:
            raise ValueError("The matrix is not square")
        matrix = self.get_matrix_memory()
        det = 1.0

        for i in range(self.n):
            # Поиск максимального элемента в текущем столбце для выбора ведущего элемента
            max_row = i
            for k in range(i, self.n):
                if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                    max_row = k

            # Если ведущий элемент ноль, то определитель равен нулю
            if matrix[max_row][i] == 0:
                return 0

            # Меняем строки местами, если нужно
            if max_row != i:
                matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
                det *= -1  # Меняем знак определителя из-за перестановки строк

            # Прямой ход метода Гаусса
            for j in range(i + 1, self.n):
                factor = matrix[j][i] / matrix[i][i]
                for k in range(i, self.n):
                    matrix[j][k] -= factor * matrix[i][k]

            # Умножаем определитель на диагональный элемент
            det *= matrix[i][i]
        return det

    def has_inverse(self) -> bool:
        """
        Определяет существование обратной матрицы для заданной квадратной матрицы.
        :return: флаг существования обратной матрицы
        """
        if self.n != self.m:
            return False
        try:
            determinant = self.get_determinant()
            if determinant == 0:
                return False
            else:
                return True
        except ValueError:
            return False


def task1_1():
    a = csr_matrix()
    a.fill_from_input()
    print(a.get_matrix_trace())

def task1_2():
    a = csr_matrix()
    a.fill_from_input()
    x, y = map(int, input().split())
    print(a[x, y])

def task2_1():
    a, b = csr_matrix(), csr_matrix()
    a.fill_from_input()
    b.fill_from_input()
    print(a + b)

def task2_2():
    a = csr_matrix()
    a.fill_from_input()
    b = float(input())
    print(a * b)

def task2_3():
    a, b = csr_matrix(), csr_matrix()
    a.fill_from_input()
    b.fill_from_input()
    print(a * b)

def task3():
    a = csr_matrix()
    a.fill_from_input()
    print(a.get_determinant())
    if a.has_inverse():
        print("да")
    else:
        print("нет")