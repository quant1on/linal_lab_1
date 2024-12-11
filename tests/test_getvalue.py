from csr_matrix import csr_matrix
import unittest

class GetGetValueTestCase(unittest.TestCase):
    def test_regular_case(self):
        matrix = [
            [1, 0, 7, 4, 1],
            [7, 4, 0, 0, 12],
            [6, 4, 0, 7, 3]
        ]
        a1 = csr_matrix()
        a1.fill_from_matrix(matrix=matrix)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                self.assertEqual(a1[i + 1, j + 1], matrix[i][j])
    
    def test_exception(self):
        a1 = csr_matrix()
        a1.fill_from_matrix(matrix=[
            [1, 0, 7, 4, 1],
            [7, 4, 0, 0, 12],
            [6, 4, 0, 7, 3]
        ])
        self.assertRaises(ValueError, csr_matrix.__getitem__, a1, (0, 0))
        self.assertRaises(ValueError, csr_matrix.__getitem__, a1, (10, 2))
        self.assertRaises(ValueError, csr_matrix.__getitem__, a1, (-2, 2))
        self.assertRaises(ValueError, csr_matrix.__getitem__, a1, (1.2, 4.3))