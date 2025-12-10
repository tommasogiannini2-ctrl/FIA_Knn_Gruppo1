import numpy as np
import pandas as pd

class KNNClassifier:

    def __init__(self, k: int):
        # Inizzializzo
        self.k = k
        self.x_training = None
        self.y_training = None

    def training_set(self, x_training, y_training):
        # assegna i dati di addestramento
        self.x_training = x_training
        self.y_training = y_training

    def distanza_euclidea(self, a, b) -> float:
        # calcola la distanza euclidea tra due punti
        # eventualmente aggiungere le altre due
        return np.sqrt(np.sum((a - b)**2))

    def classificazione(self, x_test) -> int|float:
        distanze = []
        # per ogni campione dell'insieme di test, calcola la distanza da tutti i campioni del set di training
        for x_training in self.x_training:
            euclidea = self.distanza_euclidea(x_test, x_training)
            distanze.append(euclidea)

        # creazione DataFrame di distanze e classi
        col1 = pd.Series(distanze, name = 'distanze')
        col2 = pd.Series(self.y_training, name = 'y_training')
        tab = pd.concat([col1, col2], axis = 1)

        # estrazione le etichette dei K vicini
        tab_ordinata = tab.sort_values(by = ['distanze'], ascending = True)
        indici_k = tab_ordinata.head(self.k)

        # estrazione della classe più frequente
        colonna_classi = indici_k['y_training']
        conteggio_classi = colonna_classi.value_counts()
        classe_maggiore = conteggio_classi.index[0]

        classe_1 = conteggio_classi.get(1, 0)
        prob_predetta = classe_1 / self.k

        return classe_maggiore, prob_predetta