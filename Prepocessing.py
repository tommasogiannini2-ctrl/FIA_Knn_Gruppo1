import pandas as pd
from abc import ABC
import numpy as np


class Abstract_opener(ABC):
    def open(self,dataframe_path:str)->pd.DataFrame|None:
        pass

class XLSOpener(Abstract_opener):
    def open(self,dataframe_path:str)->pd.DataFrame|None:
        self.data=pd.read_excel(dataframe_path)
        return self.data
class CSVOpener(Abstract_opener):
    def open(self, dataframe_path: str)->pd.DataFrame|None:
        # 1. Gestione di '?' come NaN
        self.data = pd.read_csv(dataframe_path, na_values=['?'])

        for col in self.data.columns:
            # errors='coerce' trasforma tutte le stringhe non valide in NaN, risolvendo il TypeError
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        return self.data

class SQLOpener(Abstract_opener):
    def open(self, dataframe_path: str)->pd.DataFrame|None:
        self.data = pd.read_sql(dataframe_path)
        return self.data

def scegli_opener(dataframe_path:str)-> SQLOpener | XLSOpener | CSVOpener:
    ext=dataframe_path.split('.')[-1]
    match ext:
        case 'csv':
            return CSVOpener()
        case 'xls':
            return XLSOpener()
        case 'sql':
            return SQLOpener()
        case _:
            raise RuntimeError(f"Unsupported file type: {ext}")

def unificaDF(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame)->pd.DataFrame:
    dataframe_unico = pd.concat([dataframe1, dataframe2], ignore_index=True)
    nuovo_nome = 'ClasseObiettivo'
    colonne_attuali = dataframe_unico.columns.tolist()
    colonne_attuali[-1] = nuovo_nome
    dataframe_unico.columns = colonne_attuali
    return dataframe_unico


class DataCsv:

    # Metodo costruttore
    def __init__(self,Opener:Abstract_opener,dataframe_path:str)->None:
        self.opener=Opener
        self.path=dataframe_path


    def load(self) -> list[pd.DataFrame]:
        self.data = self.opener.open(self.path)
        self.data = self.elimina_duplicati(self.data)

        self.data=self.elimina_outrange_features(self.data)
        self.data=self.elimina_outrange_class(self.data)

        self.data=self.elimina_classnull(self.data)
        self.classe = self.estrai_classe(self.data)
        self.data = self.elimina_features(self.data)
        self.data=self.elimina_recordnull(self.data)
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
        return dati


    def elimina_recordnull(self,dati):

        N_max_null=4
        #il thresh garantisce che chi non soddisfa la condizione di minimi valori non nulli venga eliminato
        dati = dati.dropna(thresh=len(dati.columns) - N_max_null).reset_index(drop=True)

        return dati

    #metodo eliminazione dei valori nulli (Nan e <null>)
    def elimina_nulli(self ,dati):
        #calcolo della moda di ogni colonna,scegliendo per tutti il primo valore
        for col in self.data.columns:
            mode_value = self.data[col].mode()[0]
            dati.loc[:, col] = dati.loc[:, col].fillna(mode_value)

        return dati

    def estrai_classe(self,data):
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
