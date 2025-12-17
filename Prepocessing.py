import pandas as pd
from abc import ABC
import numpy as np

class AbstractOpener(ABC):
    def open(self,dataframe_path:str)->pd.DataFrame|None:
        pass

class XLSOpener(AbstractOpener):
    def __init__(self):
        self.data= None

    def open(self, dataframe_path: str) -> pd.DataFrame | None:
        # Gestione di '?' come NaN
        self.data = pd.read_excel(dataframe_path, na_values=['?'])

        for col in self.data.columns:
            # controlla se i numeri siano decimali con il punto e non con la virgola, in caso sostituisce
            if self.data[col].dtypes == 'object':
                self.data[col] = self.data[col].str.replace(',', '.', regex=False)

            # errors='coerce' trasforma tutte le stringhe non valide in NaN, risolvendo il TypeError
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        return self.data

class CSVOpener(AbstractOpener):
    def __init__(self):
        self.data= None

    def open(self, dataframe_path: str)->pd.DataFrame|None:
        # Gestione di '?' come NaN
        self.data = pd.read_csv(dataframe_path, na_values=['?'])

        for col in self.data.columns:
            #controlla se i numeri siano decimali con il punto e non con la virgola, in caso sostituisce
            if self.data[col].dtypes == 'object':
                self.data[col]=self.data[col].str.replace(',','.',regex=False)

            # errors='coerce' trasforma tutte le stringhe non valide in NaN, risolvendo il TypeError
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        return self.data

class JSONOpener(AbstractOpener):
    def __init__(self):
        self.data= None

    def open(self, dataframe_path: str) -> pd.DataFrame | None:
        # Gestione di '?' come NaN
        self.data = pd.read_json(dataframe_path, na_values=['?'])

        for col in self.data.columns:
            # controlla se i numeri siano decimali con il punto e non con la virgola, in caso sostituisce
            if self.data[col].dtypes == 'object':
                self.data[col] = self.data[col].str.replace(',', '.', regex=False)

            # errors='coerce' trasforma tutte le stringhe non valide in NaN, risolvendo il TypeError
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        return self.data

def scegli_opener(dataframe_path:str)-> JSONOpener | XLSOpener | CSVOpener:
    ext=dataframe_path.split('.')[-1]
    match ext:
        case 'csv':
            return CSVOpener()
        case 'txt':
            return CSVOpener()
        case 'xls':
            return XLSOpener()
        case 'json':
            return JSONOpener()
        case _:
            raise RuntimeError(f"Unsupported file type: {ext}")

# Unifica il dataframe delle feature con quello della classe obbiettivo
def unificaDF(dataframe1: pd.DataFrame, dataframe2: pd.Series | pd.DataFrame)->pd.DataFrame | None:
    NOME_COLONNA_TARGET = 'classtype_v1'
    dataframe1[NOME_COLONNA_TARGET] = dataframe2
    return dataframe1

class Data:
    # Metodo costruttore
    def __init__(self,opener:AbstractOpener,dataframe_path:str)->None:
        self.opener=opener
        self.path=dataframe_path
        self.classe = None
        self.data = None

    def load(self) -> list[pd.DataFrame]:
        self.data = self.opener.open(self.path)
        self.data = self.elimina_duplicati(self.data)

        self.data=self.elimina_outrange_features(self.data)
        self.data=self.elimina_outrange_class(self.data)

        self.data=self.elimina_classnull(self.data)
        self.data = self.elimina_recordnull(self.data)
        self.classe = self.estrai_classe()
        self.data = self.elimina_features(self.data)
        self.data = self.elimina_nulli(self.data)

        print('conta quanti null ci sno per ogni colonna')
        print(self.data.isnull().sum())

        print("\n--- Informazioni sulla struttura del DataFrame ) ---")
        self.data.info()
        print("\n--- Informazioni sulla struttura della colonna classtype_v1 ) ---")
        print(self.classe.info())
        return [self.data, self.classe]

    # Metodo per eliminare duplicati
    def elimina_duplicati(self, dati):
        dati = dati.drop_duplicates()
        # Riassegna gli indici dopo l'eliminazione
        dati = dati.reset_index(drop=True)
        return dati

    #Metodo che tiene solo le features rilevanti:Clump Thickness,Uniformity of Cell Size
    #Uniformity of Cell Shape, Marginal Adhesion, Single Epithelial Cell Size
    #Bare Nuclei, Bland Chromatin, Normal Nucleoli, Mitoses
    def elimina_features(self,dati):
        features_eliminate=['Blood Pressure','Sample code number', 'Heart Rate','classtype_v1']
        for feature in features_eliminate:
            dati= dati.drop(columns=[feature],axis=1)
        return dati

    #Metodo che elimina le righe a cui corrisponde un valore nullo nella colonna classtype_v1
    def elimina_classnull(self,dati):
        target_col = 'classtype_v1'
        righe_prima = len(dati)
        # Rimuove le righe dove il valore nella colonna 'classtype_v1' è nullo (NaN)
        dati = dati.dropna(subset=[target_col]).reset_index(drop=True)
        righe_dopo = len(dati)
        if righe_dopo > righe_prima:
            print("ERRORE: le righe dopo togliere i null sono di più di quelle originali")
        return dati

    def elimina_recordnull(self,dati):
        N_max_null=4
        #il thresh garantisce che chi non soddisfa la condizione di minimi valori non nulli venga eliminato
        dati = dati.dropna(thresh=len(dati.columns) - N_max_null).reset_index(drop=True)
        return dati

    #metodo eliminazione dei valori nulli (Nan e <null>)
    def elimina_nulli(self ,dati):
        #calcolo della moda di ogni colonna,scegliendo per tutti il primo valore
        for col in dati.columns:
            mode_value = dati[col].mode()[0]
            dati.loc[:, col] = dati.loc[:, col].fillna(mode_value)
        return dati

    def estrai_classe(self):
        classe= self.data['classtype_v1']
        return classe

    def elimina_outrange_features(self, dati):
        colonna_target = 'classtype_v1'
        feature_cols = [col for col in dati.columns if col != colonna_target]
        for col in feature_cols:
         # Qui il confronto è sicuro perché il CSVOpener ha già convertito tutto in numerico.
            dati[col] = dati[col].mask((dati[col] < 1) | (dati[col] > 10), np.nan)
        return dati

    def elimina_outrange_class(self, dati):
        target_col = 'classtype_v1'
        dati[target_col] = dati[target_col].mask((dati[target_col] != 2) & (dati[target_col] != 4),np.nan)
        return dati
