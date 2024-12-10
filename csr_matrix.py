import typing as tp

# юнит-тесты напишу к среде (ко всем своим методам) - Илья
# все комментарии, сделанные через решетку - для отчета


class csr_matrix:
    def __init__(  # конструктор (будет использоваться в юнит-тестах)
        self,
        n: tp.Optional[int] = None,
        m: tp.Optional[int] = None,
        column_indices: tp.List[int] = [],
        row_pointers: tp.List[int] = [],
        values: tp.List[float] = [],
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
        self.column_indices = column_indices
        self.row_pointers = row_pointers
        self.values = values

    def fill_from_input(self) -> None:  # метод для заполнения значений из потока ввода
        """
        Заполнить матрицу разреженного-строчного типа через построчный ввод

        :return: None
        """
        self.n, self.m = map(int, input().split())
        for i in range(self.n):
            temporary_row = list(map(float, input().split()))  # считываем ввод построчно и
            # создаем из строки массив

            if len(temporary_row) > self.m:
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

    def __getitem__(self, coordinates: tp.Tuple[int, int]) -> float:  # метод для получения значения по указанным координатам
        """
        Получить значение матрицы, стоящее на ячейке с координатами coordinates

        :param coordinates: Координаты запрашиваемого значения (формат - пара чисел, Tuple)
        :return: Значение по указанным координатам
        """
        if self.n is None or self.m is None:
            raise AttributeError("Can't find value by coords for empty matrix")  # проверяем матрицу на пустоту

        if (
            (self.n < coordinates[0] or self.m < coordinates[1])
            or (coordinates[0] < 0 or coordinates[1] < 0)
            or len(coordinates) != 2
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
        return 0.0  # если нет - значит, на этом месте стоит 0

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
            trace += self[i, i]  # складываем все элементы на главной диагонали, используя метод get_value()

        return trace

    def __add__(self, other) -> tp.Self:  # перегрузка оператора сложения (для реализации метода сложения матриц)
        pass

    def __mul__(
        self, other
    ) -> tp.Self:  # перегрузка оператора умножения (для реализации метода умножения матриц друг на друга и домножения на скаляр)
        pass

    def get_determinant(self) -> float:  # метод для получения детерминанта
        pass
