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


