import pandas as pd
import numpy as np
from development import *

class ValidationStrategy:
    """
    Classe contenente tutti i metodi necessari per effettuare le validation strategy.
    """
    def __init__(self, data):
        self.dati = data

    def Kfold(self, k_prove: int) -> list[list[pd.DataFrame]]:
        """
        Funzione che implementa il metodo di validazione Kfold.
        Il dataframe originale viene diviso in K parti.
        Verranno effettuati K esperimenti dove si avrà un blocco di test
        e tutti gli altri saranno di training
        :param k_prove: numero di divisioni da effettuare
        :return: lista di liste contenenti dataframe di training e test
        """
        #divido il set in k parti
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
            #returna una lista di k coppie training e test
        return lista

    def RandomSubsampling(self, n, p) -> list[list[pd.DataFrame]]:
        """
        Funzione che implementa il metodo di validazione Random Subsampling.
        Il dataframe originale viene diviso in 2 parti grazie alla percentuale p.
        :param n: numero di prove da effettuare
        :param p: percentuale di divisione in training e test
        :return: lista di liste contenenti dataframe di training e test
        Questo codice viene utilizzato anche per la validazione Holdout,
        essa infatti è semplicemente un Random Subsampling ripetuto una volta sola
        """
        #ripetizione di n volte l'estrazione del training set che ha una percentuale p
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

class Evaluation:
    """
    Classe per calcolare metriche di valutazione.
    """
    def __init__(self, classe_vera, classe_predetta):
        self.classe_vera = classe_vera
        self.classe_predetta = classe_predetta
        self.TP = None
        self.TN = None
        self.FP = None
        self.FN = None
        self.P = None
        self.N = None
        self.Total= None
        self.confusion_matrix = None

        if self.classe_predetta is not None:
            self.matrice_confusione()

    def matrice_confusione(self):
        """
        Calcola gli elementi della matrice di confusione.
        """
        self.TP = np.sum((self.classe_vera == 4) & (self.classe_predetta == 4))
        self.TN = np.sum((self.classe_vera == 2) & (self.classe_predetta == 2))
        self.FP = np.sum((self.classe_vera == 2) & (self.classe_predetta == 4))
        self.FN = np.sum((self.classe_vera == 4) & (self.classe_predetta == 2))

        #calcolo dei positivi totali
        self.P = self.TP + self.FN
        #calcolo dei negativi totali
        self.N = self.TN + self.FP
        #numero casi totali
        self.Total = len(self.classe_vera)

        self.confusion_matrix = np.array([[self.TN, self.FP], [self.FN, self.TP]])
        return self.confusion_matrix

    def accuracy_rate(self):
        return (self.TP + self.TN) / self.Total

    def error_rate(self):
        return (self.FP + self.FN) / self.Total

    def sensitivity(self):
        if self.P == 0:
            return np.nan
        return self.TP / self.P

    def specificity(self):
        if self.N == 0:
            return np.nan
        return self.TN / self.N

    def geometric_mean(self):
        sensitivity = self.sensitivity()
        specificity = self.specificity()
        return np.sqrt(sensitivity * specificity)

    def roc_curve(self, classe_vera, thresholds, prob_predette):
        """
         Metodo che calcola la curva ROC.
         Si ordinano i campioni per probabilità decrescente.
         Si fanno scorrere le soglie dal valore più alto al più basso.
         Per ogni sogli si calcola TPR e FPR e si uniscono i punti
        :param classe_vera: vettore che contiene la ground truth associata a un record
        :param thresholds: vettore che contiene le soglie da confrontare
        :param prob_predette: vettore che contiene la probabilità della classe predetta
        :return: due liste di True Positive Rate e False Positive Rate
        """
        TPR = []
        FPR = []

        classe_vera = np.where(np.array(classe_vera) == 4, 1, 0)
        prob_predette = np.array(prob_predette)

        for threshold in thresholds:
            classe_predetta_soglia = (prob_predette >= threshold).astype(int)
            # ricalcolo degli elementi della matrice di confusione per la soglia corrente
            TP = np.sum((classe_vera == 1) & (classe_predetta_soglia == 1))
            TN = np.sum((classe_vera == 0) & (classe_predetta_soglia == 0))
            FP = np.sum((classe_vera == 0) & (classe_predetta_soglia == 1))
            FN = np.sum((classe_vera == 1) & (classe_predetta_soglia == 0))

            P = TP + FN
            N = TN + FP

            TPR_p = TP / P if P > 0 else 0
            FPR_p = FP / N if N > 0 else 0

            TPR.append(TPR_p)
            FPR.append(FPR_p)

        # Ordiniamo per FPR per assicurarci che il grafico e l'AUC siano corretti
        indici = np.argsort(FPR)
        FPR_sorted = np.array(FPR)[indici]
        TPR_sorted = np.array(TPR)[indici]

        return FPR_sorted, TPR_sorted

    def area_under_the_curve(self, fpr, tpr):
        """
        Metodo che calcola l'area al di sotto della curva ROC utilizzando il metodo del trapezio
        :param fpr: vettore del False positive rate
        :param tpr: vettore del True positive rate
        :return: l'area sotto la curva (numero reale)
        """
        # calcolo dell'AUC utilizzando il metodo del trapezio
        fpr = np.array(fpr)
        tpr = np.array(tpr)

        auc = 0.0
        for i in range(len(fpr) - 1):
            base = fpr[i + 1] - fpr[i]
            altezza_media = (tpr[i] + tpr[i + 1]) / 2.0
            auc += base * altezza_media
        return auc

