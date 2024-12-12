from csr_matrix import csr_matrix
import unittest

class GetHasInverseTestCase(unittest.TestCase):
    def test_true_case(self):

        a = csr_matrix()

        a.fill_from_matrix(matrix=[
            [12, 3, 0, 12, 3, 2],
            [7, 0, 4, 5, 3, 6],
            [6, 0, 5, 4, 2, 1],
            [2, 3, 2, 3, 8, 5],
            [2, 5, 6, 0, 2, 4],
            [5, 0, 3, 5, 3, 2]
        ])

        self.assertTrue(a.has_inverse())

    def test_false_case(self):
        
        a = csr_matrix()

        a.fill_from_matrix(matrix=[
            [12, 32, 124, 0],
            [54, 34, 31, 23],
            [0, 0, 0, 0],
            [123, 54, 5, 3]
        ])

        self.assertFalse(a.has_inverse())

    def test_nonsquare(self):

        a = csr_matrix()

        a.fill_from_matrix(matrix=[
            [1, 2, 3],
            [4, 5, 6]
        ])

        self.assertFalse(a.has_inverse())