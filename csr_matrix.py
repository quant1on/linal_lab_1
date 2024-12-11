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
        :param colummn_indices: Массив индексов столбцов каждого элемента values
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
        self.n, self.m = map(int, input().split())
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
                    self.values.append(temporary_row[j])  # записываем только ненулевые значения
                    self.column_indices.append(j)

        self.row_pointers.append(len(self.values))  # фиксируем конец последней строки

    def fill_from_matrix(self, matrix: tp.List[tp.List[float]]) -> None:
        """
        Заполнить данную csr_matrix на основе матрицы в обычном представлении

        :param matrix: Матрица, из которой нужно взять значения
        :return: None
        """
        # буквально аналог функции fill_from_input, но только чтобы для тестов
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

        if (self.n < coordinates[0] or self.m < coordinates[1]) or (coordinates[0] < 0 or coordinates[1] < 0) or len(coordinates) != 2:
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

    # комменты для отчета добавлю позже
    def __add__(self, other) -> tp.Self:  # перегрузка оператора сложения (для реализации метода сложения матриц)
        if not isinstance(other, csr_matrix):
            raise AttributeError("Can't sum a matrix and a non-matrix type")

        if self.n != other.n or self.m != other.m:
            raise AttributeError("Can't sum matrices of different sizes")

        mtrx_sum = csr_matrix(n=self.n, m=self.m)
        for i in range(self.n):
            row_start_1, row_end_1, row_start_2, row_end_2 = (
                self.row_pointers[i],
                self.row_pointers[i + 1],
                other.row_pointers[i],
                other.row_pointers[i + 1],
            )
            i_1 = row_start_1
            i_2 = row_start_2
            mtrx_sum.row_pointers.append(len(mtrx_sum.values))
            for j in range(self.m):
                while i_1 < row_end_1 - 1 and j > self.column_indices[i_1]:
                    i_1 += 1
                while i_2 < row_end_2 - 1 and j > other.column_indices[i_2]:
                    i_2 += 1
                sum_ = 0.0
                if i_1 < row_end_1 and self.column_indices[i_1] == j:
                    sum_ += self.values[i_1]
                if i_2 < row_end_2 and other.column_indices[i_2] == j:
                    sum_ += other.values[i_2]
                if sum_ != 0:
                    mtrx_sum.values.append(sum_)
                    mtrx_sum.column_indices.append(j)

        mtrx_sum.row_pointers.append(len(mtrx_sum.values))

        return mtrx_sum

    # комменты для отчета добавлю позже
    def __mul__(
        self, other
    ) -> tp.Self:  # перегрузка оператора умножения (для реализации метода умножения матриц друг на друга и домножения на скаляр)
        if not isinstance(other, (csr_matrix, float, int)):
            raise AttributeError("Invalid multiplication")

        if isinstance(other, (float, int)):  # умножение на скаляр
            mtrx_mul = csr_matrix(n=self.n, m=self.m)
            if other == 0.0:
                mtrx_mul.row_pointers = [0] * (self.n + 1)
            else:
                mtrx_mul.column_indices = self.column_indices.copy()
                mtrx_mul.row_pointers = self.row_pointers.copy()
                mtrx_mul.values = self.values.copy()
                for i in range(len(self.values)):
                    mtrx_mul.values[i] *= other

        else:  # умножение матриц
            if self.m != other.n:
                raise AttributeError("Can't multiply matrices of non-compatible sizes")

            mtrx_mul = csr_matrix(n=self.n, m=other.m)

            for i in range(self.n):
                mtrx_mul.row_pointers.append(len(mtrx_mul.values))
                for j in range(other.m):
                    mul_ = 0.0

                    for k in range(self.m):
                        mul_ += self[i + 1, k + 1] * other[k + 1, j + 1]  # очень неоптимизированный способ, надо переделать

                    if mul_ == 0:
                        continue
                    else:
                        mtrx_mul.values.append(mul_)
                        mtrx_mul.column_indices.append(j)
            mtrx_mul.row_pointers.append(len(mtrx_mul.values))

        return mtrx_mul

    def __rmul__(self, other) -> tp.Self:  # метод обратного умножения для коммутативности умножения на скаляр
        if not isinstance(other, (float, int)):
            raise AttributeError("Invalid multiplication")

        return self.__mul__(other)

    def __repr__(self) -> str:  # метод для вывода экземпляра класса в обычном матричном виде (для относительно малых матриц)
        if self.n is None or self.m is None:
            return ""
        matrix = ""

        for i in range(self.n):
            for j in range(self.m):
                matrix += f"{self[i + 1, j + 1]} "
            matrix += "\n"

        return matrix

    def __eq__(self, other):
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

    def get_determinant(self) -> float:  # метод для получения детерминанта
        pass
