from csr_matrix import csr_matrix
import unittest

class GetMultiplicationTestCase(unittest.TestCase):
    def test_regular_mul(self):
        a1, a2, a_mul = csr_matrix(), csr_matrix(), csr_matrix()

        a1.fill_from_matrix(matrix=[
            [1, 0, 3, 0, 6, 0, 7],
            [0, 8, 0, 10, 0, 12, 0],
            [13, 0, 0, 16, 17, 0, 19],
            [0, 21, 22, 0, 24, 0, 26],
            [27, 0, 29, 0, 0, 32, 0],
            [0, 34, 0, 36, 37, 0, 39]
        ])

        a2.fill_from_matrix(matrix=[
            [6, 0, 4, 0, 2, 0],
            [0, 11, 0, 9, 0, 7],
            [18, 0, 0, 15, 0, 13],
            [0, 23, 22, 0, 20, 19],
            [30, 0, 0, 27, 26, 0],
            [0, 35, 0, 33, 0, 31],
            [36, 0, 34, 0, 32, 0]
        ])

        a_mul.fill_from_matrix(matrix=[
            [492, 0, 242, 207, 382, 39],
            [0, 738, 220, 468, 200, 618],
            [1272, 368, 1050, 459, 1396, 304],
            [2052, 231, 884, 1167, 1456, 433],
            [684, 1120, 108, 1491, 54, 1369],
            [2514, 1202, 2118, 1305, 2930, 922]
        ])

        self.assertEqual(a1 * a2, a_mul)
    
    def test_scalar_mul(self):
        a1, a_mul = csr_matrix(), csr_matrix()
        a1.fill_from_matrix(matrix=[
            [1, 2, 3, 4, 5],
            [0, 9, 6, 4, 9],
            [0, 7, 5, 8, 0]
        ])

        a2 = 3

        a_mul.fill_from_matrix(matrix=[
            [3, 6, 9, 12, 15],
            [0, 27, 18, 12, 27],
            [0, 21, 15, 24, 0]
        ])

        self.assertEqual(a1 * a2, a_mul)
        self.assertEqual(a2 * a1, a_mul)
    
    def test_size_protection(self):
        a1, a2 = csr_matrix(), csr_matrix()

        a1.fill_from_matrix(matrix=[
            [1, 2, 3, 4, 5],
            [0, 9, 6, 4, 9],
            [0, 7, 5, 8, 0]
        ])

        a2.fill_from_matrix(matrix=[
            [1, 2, 3],
            [4, 5, 6]
        ])

        self.assertRaises(AttributeError, csr_matrix.__mul__, a1, a2)

    def test_type_protection(self):
        a1 = csr_matrix()

        a1.fill_from_matrix(matrix=[
            [1, 2, 3],
            [4, 5, 6]
        ])

        a2 = "123"
        self.assertRaises(AttributeError, csr_matrix.__mul__, a1, a2)

    def test_zero_scalar(self):
        a1 = csr_matrix()

        a1.fill_from_matrix(matrix=[
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        a2 = 0

        self.assertEqual(a1 * a2, csr_matrix(n=2, m=3, row_pointers=[0, 0, 0]))