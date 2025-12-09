import pandas as pd
import numpy as np

class EvaluationStrategy:
    def __init__(self, data):
        self.dati = data

    def Kfold(self, k_prove: int) -> list[list[pd.DataFrame]]:
        dimensione_parte = len(self.dati)//k_prove
        lista = []
        for i in range(k_prove):
            # Calcola l'indice di inizio e fine
            inizio = i * dimensione_parte
            # Per l'ultima parte, assicurati di prendere tutte le righe rimanenti
            fine = (i + 1) * dimensione_parte if i < k_prove - 1 else len(self.dati)
            lista.append(self.dati[inizio:fine])

            # Estrai il sotto-DataFrame
            test = self.dati.iloc[inizio:fine]
            training = self.dati.drop(test.index)
            lista.append([training,test])
        return lista


    def RandomSubsampling(self, n, p) -> list[list[pd.DataFrame]]:
        lista =[]
        len_col = self.dati["Mitoses"].shape[0]
        indice_divisione = np.floor(p * len_col)

        for i in range(n):
            self.dati = self.dati.sample(frac=1)
            training = self.dati.head(indice_divisione)
            test = self.dati.tail(indice_divisione)
            lista.append([training,test])
        #ritorna una lista di liste dove in ogni lista interna c'è training e test
        return lista

class Metriche:

    def __init__(self, classe_vera, classe_predetta):
        self.classe_vera = classe_vera
        self.classe_predetta = classe_predetta

        if self.classe_predetta is not None:
            self.matrice_confusione()

    def matrice_confusione(self):
        self.TP = np.sum((self.classe_vera == 1) & (self.classe_predetta == 1))
        self.TN = np.sum((self.classe_vera == 0) & (self.classe_predetta == 0))
        self.FP = np.sum((self.classe_vera == 0) & (self.classe_predetta == 1))
        self.FN = np.sum((self.classe_vera == 1) & (self.classe_predetta == 0))

        self.P = self.TP + self.FN
        self.N = self.TN + self.FP
        self.Total = len(self.classe_vera)

    def accuracy_rate(self):
        return (self.TP + self.TN) / self.Total

    def error_rate(self):
        return (self.FP + self.FN) / self.Total

    def sensitivity(self):
        return self.TP / self.P

    def specificity(self):
        return self.TN / self.N

    def geometric_mean(self):
        sensitivity = self.sensitivity()
        specificity = self.specificity()
        return np.sqrt(sensitivity * specificity)

    def area_under_the_curve(self):
        pass


