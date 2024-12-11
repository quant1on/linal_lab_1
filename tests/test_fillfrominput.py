from csr_matrix import csr_matrix
import unittest
from unittest.mock import patch

class GetFillFromInputTestCase(unittest.TestCase):
    @patch('builtins.input', side_effect = ['4 5',
                                            '3 0 1 2 4',
                                            '5 6 -9 3 1',
                                            '4 0 0 0 2',
                                            '5 7 0 4 -1'])
    def test_regular_case(self, mock):
        a, a_res = csr_matrix(), csr_matrix(n=4, m=5,
                                            column_indices=[0, 2, 3, 4, 0, 1, 2, 3, 4, 0, 4, 0, 1, 3, 4],
                                            row_pointers=[0, 4, 9, 11, 15],
                                            values=[3, 1, 2, 4, 5, 6, -9, 3, 1, 4, 2, 5, 7, 4, -1])
        a.fill_from_input()
        self.assertEqual(a, a_res)
    
    @patch('builtins.input', side_effect=['2 3',
                                          '2 3 1',
                                          '3 0 0 1'])
    def test_exception(self, mock):
        a = csr_matrix()
        self.assertRaises(ValueError, csr_matrix.fill_from_input, a)
