import typing as tp

class csr_matrix:
    def __init__( #конструктор (будет использоваться в юнит-тестах)
            self,
            n: tp.Optional[int] = None,
            m: tp.Optional[int] = None,
            column_indices: tp.List[int] = [],
            row_pointers: tp.List[int] = [],
            values: tp.List[float] = []
            ) -> None:
        self.n = n
        self.m = m
        self.column_indices = column_indices
        self.row_pointers = row_pointers
        self.values = values

    def fill_from_input(self) -> None: #метод для заполнения значений из потока ввода
        pass
        
    def get_value(self, coordinates: tp.Tuple[int, int]) -> float: #метод для получения значения по указанным координатам
        pass

    def get_matrix_trace(self) -> float: # метод для получения следа матрицы
        pass
    
    def __add__(self, other) -> tp.Self: #перегрузка оператора сложения (для реализации метода сложения матриц)
        pass

    def __mul__(self, other) -> tp.Self: #перегрузка оператора умножения (для реализации метода умножения матриц друг на друга и домножения на скаляр)
        pass

    def get_determinant(self) -> float: #метод для получения детерминанта 
        pass
