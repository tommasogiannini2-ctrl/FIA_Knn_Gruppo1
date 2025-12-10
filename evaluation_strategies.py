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
            # Estrai il sotto-DataFrame
            test = self.dati.iloc[inizio:fine]
            training = self.dati.drop(test.index)
            lista.append([training,test])
        return lista


    def RandomSubsampling(self, n, p) -> list[list[pd.DataFrame]]:
        lista =[]
        len_col = len(self.dati)
        indice_divisione = int(np.floor(p * len_col))

        for i in range(n):
            dati_shuffle = self.dati.sample(frac=1)
            training = dati_shuffle.iloc[:indice_divisione]
            test = dati_shuffle.iloc[indice_divisione:]
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

    def area_under_the_curve(self, classe_vera, thresholds, prob_predette):
        TPR = []
        FPR = []

        classe_vera = np.array(classe_vera)
        prob_predette = np.array(prob_predette)

        for threshold in thresholds:
            classe_predetta_soglia = (prob_predette >= threshold)
            TP = np.sum((classe_vera == 1) & (classe_predetta_soglia == 1))
            TN = np.sum((classe_vera == 0) & (classe_predetta_soglia == 0))
            FP = np.sum((classe_vera == 0) & (classe_predetta_soglia == 1))
            FN = np.sum((classe_vera == 1) & (classe_predetta_soglia == 0))

            P = TP + FN
            N = TN + FP

            TPR_p = TP / P
            FPR_p = FP / N

            TPR.append(TPR_p)
            FPR.append(FPR_p)

        sort_idx = np.argsort(FPR)
        FPR_sorted = np.array(FPR)[sort_idx]
        TPR_sorted = np.array(TPR)[sort_idx]

        auc = 0
        for i in range(len(FPR_sorted) - 1):
            delta_FPR = FPR_sorted[i + 1] - FPR_sorted[i]
            avg_TPR = (TPR_sorted[i] + TPR_sorted[i + 1]) / 2
            area_trapezio = delta_FPR * avg_TPR
            auc += area_trapezio

        return auc
