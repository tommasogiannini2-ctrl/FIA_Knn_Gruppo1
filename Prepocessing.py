from unittest import case

#VERSIONE DI TOMMASO
import pandas as pd
from abc import ABC

class Abstract_opener(ABC):
    def open(self,dataframe_path:str)->pd.DataFrame|None:
        pass

class XLS_Opener():
    def open(self,dataframe_path:str)->pd.DataFrame|None:
        self.data=pd.read_excel(dataframe_path)
        return self.data
class CSV_opener():
    def open(self, dataframe_path: str)->pd.DataFrame|None:
        self.data = pd.read_csv(dataframe_path)
        return self.data

class SQL_opener():
    def open(self, dataframe_path: str)->pd.DataFrame|None:
        self.data = pd.read_sql(dataframe_path)
        return self.data

def open(dataframe_path:str)->Abstract_opener:
    ext=dataframe_path.split('.')[-1]
    match ext:
        case 'csv':
            return CSV_opener()
        case 'xls':
            return XLS_Opener()
        case 'sql':
            return SQL_opener()
        case _:
            raise RuntimeError(f"Unsupported file type: {ext}")


class data_csv():

    # Metodo costruttore
    def __init__(self,Opener:Abstract_opener):
        self.opener=Opener


    def load(self, dataframe_path:str)-> pd.DataFrame:
       self.data = self.opener.open(dataframe_path)
       self.data = self.elimina_duplicati(self.data)
       self.data=self.elimina_features(self.data)
       self.data=self.elimina_nulli(self.data)
       print("\n--- Informazioni sulla struttura del DataFrame ) ---")
       self.data.info()


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
        features_eliminate=['Blood Pressure','Sample code number', 'Heart Rate']
        for feature in features_eliminate:
            dati= dati.drop(columns=[feature],axis=1)

        return dati

    def elimina_classnull(self,dati):

        return dati

    def elimina_recordnull(self,dati):

        return dati

    #metodo eliminazione dei valori nulli (Nan e <null>)
    def elimina_nulli(self ,dati):
        #decido d sostituire i valori mancanti con la moda di ogni colonna, perchè la ritengo più segnificativa
        # rispetto a media e mediana,perchè non sapendo se una cellula è tumorale o no prendo il valore che
        # statisticamente si ripete di più

        #calcolo della moda di ogni colonna,scegliendo per tutti il primo valore
        for col in self.data.columns:
            mode_value = self.data[col].mode()[0]
            dati.loc[:, col] = dati.loc[:, col].fillna(mode_value)

        #verifica che tutti i Nan siano eliminati
        if(dati.isnull().sum()==0).all():
            print('nessun valore nullo')
        else:
            print('valori nullo')
        return dati



# Esecuzione

dati = data_csv()
dati=dati.load('./dati/version_1.csv')
print(dati)


#fai la factory
#se più di due null sul record eliminalo
#valuta se eliminare il record dove il class type è nullo


