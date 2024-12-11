from csr_matrix import csr_matrix
import unittest

class GetGetMatrixTraceTestCase(unittest.TestCase):
    def test_regular_case(self):
        a = csr_matrix()
        a.fill_from_matrix(matrix=[
            [1, 2, 0, 6, 4],
            [4, 3, 6, 0, 6],
            [5, 2, 0, 8, 5],
            [0, 8, 0, 0, 1],
            [0, 3, 2, 0, -1]
        ])

        self.assertEqual(a.get_matrix_trace(), 3)
    
    def test_exception(self):
        a = csr_matrix()
        a.fill_from_matrix(matrix=[
            [1, 2, 0, 1],
            [3, 1, 3, 2],
            [5, 3, 1, 2]
        ])
        self.assertRaises(AttributeError, csr_matrix.get_matrix_trace, a)