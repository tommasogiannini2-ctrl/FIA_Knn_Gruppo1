import pandas as pd
import numpy as np
from developement import *

class ValidationStrategy:
    def __init__(self, data):
        self.dati = data

    def Kfold(self, k_prove: int) -> list[list[pd.DataFrame]]:
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

    def area_under_the_curve(self, classe_vera, thresholds, prob_predette):
        TPR = []
        FPR = []

        classe_vera = np.where(np.array(classe_vera) == 4,1,0)
        prob_predette = np.array(prob_predette)

        # itera su ogni soglia per calcolare true positive rate e false positive rate
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

        # ordina i punti in base a FPR
        indici = np.argsort(FPR)
        FPR_sorted = np.array(FPR)[indici]
        TPR_sorted = np.array(TPR)[indici]

        # calcolo dell'AUC utilizzando il metodo del trapezio
        auc = 0
        for i in range(len(FPR_sorted) - 1):
            FPR_trap = FPR_sorted[i + 1] - FPR_sorted[i]
            TPR_trap = (TPR_sorted[i] + TPR_sorted[i + 1]) / 2
            area_trapezio = FPR_trap * TPR_trap
            auc += area_trapezio

        return auc

def calcolo_metriche(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame)-> dict | None:
    classificatore = KNNClassifier(dataframe1, dataframe2)
    classificatore.separatore()
    k_ottimale = classificatore.knn_k_ottimale()
    y_predette, prob_class_4 = classificatore.restituzione_classepredetta(classificatore.x_test.values)
    classe_vera = np.round(classificatore.y_test.values).astype(int)
    classe_predetta = np.round(y_predette).astype(int)

    evaluation = Evaluation(classe_vera, classe_predetta)
    evaluation.matrice_confusione()

    thresholds = np.linspace(1.0, 0.0, 1001)
    auc = evaluation.area_under_the_curve(classificatore.y_test.values, thresholds, prob_class_4)

    risultati = {
        'k': k_ottimale,
        'accuracy': evaluation.accuracy_rate(),
        'error_rate': evaluation.error_rate(),
        'sensitivity': evaluation.sensitivity(),
        'specificity': evaluation.specificity(),
        'geometric_mean': evaluation.geometric_mean(),
        'auc': auc
    }
    return risultati

def calcolo_media_stddev_metriche(lista_ris: list)-> pd.DataFrame | None:
    risultati = pd.DataFrame(lista_ris)
    metriche = risultati.columns.drop('k')

    # Medie delle metriche
    lista_metrica = []
    lista_media = []
    lista_devstd = []

    # Calcola la media e la deviazione standard per ogni colonna
    for colonna in metriche:
        media = risultati[colonna].mean()
        deviazione_standard = risultati[colonna].std()

        lista_metrica.append(colonna)
        lista_media.append(media)
        lista_devstd.append(deviazione_standard)

    k_medio = risultati['k'].mode()

    risultati_finali = pd.DataFrame({
        'Metrica': lista_metrica,
        'Media': lista_media,
        'Deviazione Standard': lista_devstd
    })

    risultati_finali.loc[-1] = ['K Ottimale Medio', k_medio, np.nan]
    risultati_finali.index = risultati_finali.index + 1
    risultati_finali = risultati_finali.sort_index().reset_index(drop=True)

    return risultati_finali