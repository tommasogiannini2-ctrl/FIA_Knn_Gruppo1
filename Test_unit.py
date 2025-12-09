import unittest
from Prepocessing import *


class TestDataPreprocessing(unittest.TestCase):
    def test_path_not_found_exception_raised(self):
        """
        Verifica che se viene fornito un path errato viene alzata la giusta eccezione
        """
        non_existent_path = "path/che/non/esiste/Prova1.csv"
        p = DataCsv(opener, non_existent_path)
        with self.assertRaises(FileNotFoundError):
            p.load()


if __name__ == "__main__":
    unittest.main()