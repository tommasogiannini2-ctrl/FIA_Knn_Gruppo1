import pandas as pd
import numpy as np

class EvaluationStrategy:
    def __init__(self, data):
        self.dati = data

    def Kfold(self, dati: pd.DataFrame,):
        pass

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


