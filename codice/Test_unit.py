import unittest
from Prepocessing import *
from validation_evaluation_strategies import *

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        # Inizializzazione DataFrame di prova
        data = {
            'ID_Paziente': [1001, 1002, 1003, 1004, 1005],  # Tutti gli ID sono diversi
            'Nome': ['Mario', 'Luisa', 'Giovanni', 'Anna', 'Paolo'],
            'Età': [45, 62, 33, 78, 55],
            'Trattamento': ['Farmaco A', 'Farmaco B', 'Placebo', 'Farmaco A', 'Farmaco C'],
            'Pressione_Sistolica': [135, 160, 110, 155, 128],
            'Data_Ricovero': pd.to_datetime(['2024-10-01', '2024-09-20', '2024-11-05', '2024-09-01', '2024-10-15'])
        }

        self.df_prova = pd.DataFrame(data)

    def test_path_not_found_exception_raised(self):
        """
        Verifica che se viene fornito un path errato viene alzata la giusta eccezione
        """
        opener =  CSVOpener()
        non_existent_path = "path/che/non/esiste/Prova1.csv"
        p = Data(opener, non_existent_path)
        with self.assertRaises(FileNotFoundError):
            p.load()

    def test_random_subsampling_intersection_empy(self):
        """
        Verifica che i dataframe di test e training hanno intersezione vuota
        """
        df = ValidationStrategy(self.df_prova)
        l = df.RandomSubsampling(1, 0.8)
        coppiadf = l[0]
        training, test = coppiadf[0], coppiadf[1]
        all_columns = training.columns.tolist()
        intersezione_df = pd.merge(training,test,on=all_columns,how='inner',indicator=False)

        self.assertTrue(intersezione_df.empty,
                        f"L'intersezione non è vuota. Trovate {len(intersezione_df)} righe comuni.")

    def test_kfold_intersection_empy(self):
        """
        Verifica che i dataframe di test e training hanno intersezione vuota, per le due divisioni.
        Verifica inoltre che i due dataframe di test hanno intersezione vuota
        """
        df = ValidationStrategy(self.df_prova)
        l = df.Kfold(2)

        coppia1 = l[0]
        training1, test1 = coppia1[0], coppia1[1]

        coppia2 = l[1]
        training2, test2 = coppia2[0], coppia2[1]

        all_columns = training1.columns.tolist()

        intersezione_df_1 = pd.merge(training1, test1, on=all_columns, how='inner', indicator=False)
        self.assertTrue(intersezione_df_1.empty, "Il primo fold ha righe in comune (Training1 vs Test1).")

        intersezione_df_2 = pd.merge(training2, test2, on=all_columns, how='inner', indicator=False)
        self.assertTrue(intersezione_df_2.empty, "Il secondo fold ha righe in comune (Training2 vs Test2).")

        intersezione_test_sets = pd.merge(test1, test2, on=all_columns, how='inner', indicator=False)
        self.assertTrue(intersezione_test_sets.empty, "I Test Set dei diversi fold non sono disgiunti.")

class TestMetrics(unittest.TestCase):
    def setUp(self):
        classe_vera = np.array([4] * 10 + [2] * 10)  # 10 Positivi, 10 Negativi
        classe_predetta = np.array([4] * 7 + [2] * 3 + [2] * 7 + [4] * 3)  # 7 TP, 3 FN, 7 TN, 3 FP
        self.eval_test = Evaluation(classe_vera, classe_predetta)

    def test_matrice_di_confusione(self):
        # Setup
        assert self.eval_test.TP == 7
        assert self.eval_test.FN == 3
        assert self.eval_test.TN == 7
        assert self.eval_test.FP == 3
        assert self.eval_test.P == 10
        assert self.eval_test.N == 10
        assert self.eval_test.Total == 20

    def test_accuracy(self):
        #accuracy_attesa = (7 + 7) / 20  14/20 = 0.7
        assert self.eval_test.accuracy_rate() == 0.7

    def test_sensitivity(self):
        #sensitivity_attesa = 7 / 10  # TP / P
        assert self.eval_test.sensitivity() == 0.7

    def test_specificity(self):
        #specificity_attesa = 7 / 10  # TN / N
        assert self.eval_test.specificity() == 0.7

    def test_geometric_mean(self):
        #geometric_mean_attesa = np.sqrt(0.7 * 0.7)  # 0.7
        assert np.isclose(self.eval_test.geometric_mean(), 0.7)


if __name__ == "__main__":
    unittest.main()