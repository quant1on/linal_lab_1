from csr_matrix import csr_matrix
import unittest

class GetAdditionTestCase(unittest.TestCase):
    def test_regular_sum(self):
        a1, a2, a_sum = csr_matrix(), csr_matrix(), csr_matrix()

        a1.fill_from_matrix(matrix=[
            [0, 3, 0, -9, 9, 5, 0],
            [4, 0, -7, 76, 4, 0, 0],
            [5, 6, -34, 0, 0, 0, 0],
            [4, 9, 0, 0, 0, -6, 9],
            [0, 0, 0, 0, 0, 0, 0]
        ])

        a2.fill_from_matrix(matrix=[
            [0, 0, 0, 0, 0, 0, 0],
            [6, 8, 0, 0, 0, 3, 9],
            [8, 7, 4, 3, 0, 0, 0],
            [0, 0, 6, 5, 4, 9, 3],
            [3, 9, 0, 0, 0, 0, 0]
        ])

        a_sum.fill_from_matrix(matrix=[
            [0, 3, 0, -9, 9, 5, 0],
            [10, 8, -7, 76, 4, 3, 9],
            [13, 13, -30, 3, 0, 0, 0],
            [4, 9, 6, 5, 4, 3, 12],
            [3, 9, 0, 0, 0, 0, 0]
        ])

        self.assertEqual(a1 + a2, a_sum)
        self.assertEqual(a2 + a1, a_sum)
    
    def test_empty_sum(self):
        a1, a2, a_sum = csr_matrix(), csr_matrix(), csr_matrix()
        a1.fill_from_matrix(matrix=[
            [1, 54, 23, 0, 0, 23, 1],
            [6, 0, 7, 0, -5, 3, -8],
            [7, 5, 5, 0, 0, 0, 7],
            [4, 2, 3, 0, 0, 0, 0],
            [6, 5, 4, 5, 0, 0, 0],
            [7, 5, 0, 0, 0, 5, 4]
        ])

        a2.fill_from_matrix(matrix=[
            [-1, -54, -23, 0, 0, -23, -1],
            [-6, 0, -7, 0, 5, -3, 8],
            [-7, -5, -5, 0, 0, 0, -7],
            [-4, -2, -3, 0, 0, 0, 0],
            [-6, -5, -4, -5, 0, 0, 0],
            [-7, -5, 0, 0, 0, -5, -4]
        ])
        
        a_sum.fill_from_matrix(matrix=[
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])

        self.assertEqual(a2 + a1, a_sum)

    def test_size_protection(self):
        a1, a2 = csr_matrix(), csr_matrix()

        a1.fill_from_matrix(matrix=[
            [1, 2, 3, 4],
            [3, 4, 5, 0],
            [7, 5, 7, 0]
        ])

        a2.fill_from_matrix(matrix=[
            [1, 3, 5, 3],
            [5, 6, 8, 8]
        ])

        self.assertRaises(AttributeError, csr_matrix.__add__, a1, a2)
        self.assertRaises(AttributeError, csr_matrix.__add__, a2, a1)

    def test_type_protection(self):
        a1 = csr_matrix()
        a2 = "213"

        a1.fill_from_matrix(matrix=[
            [0, 0, 0, 0], 
            [0, 0, 0, 0]
        ])

        self.assertRaises(AttributeError, csr_matrix.__add__, a1, a2)
        self.assertRaises(AttributeError, csr_matrix.__add__, a2, a1)