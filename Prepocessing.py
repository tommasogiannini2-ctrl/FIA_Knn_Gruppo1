
#VERSIONE DI TOMMASO
import pandas as pd

class data_csv:

    # Metodo costruttore
    def __init__(self, dataframe_path:str):
        # Inizializza i dati leggendo il file csv
       try:
           self.data = pd.read_csv(dataframe_path)
           print(f"File '{dataframe_path}' estratto con successo nel DataFrame 'df'.\n")
       except FileNotFoundError:
           print(
               f"ERRORE: Il file '{dataframe_path}' non è stato trovato. Assicurati che sia nella stessa cartella dello script Python o che il percorso sia corretto.")
       except Exception as e:
           print(f"Si è verificato un errore durante la lettura del file: {e}")

       self.data = self.elimina_duplicati(self.data)
       self.data=self.elimina_features(self.data)
       self.data=self.elimina_nulli(self.data)
       print("\n--- Informazioni sulla struttura del DataFrame ) ---")
       self.data.info()


    # Metodo per eliminare duplicati
    def elimina_duplicati(self, dati: pd.DataFrame) -> pd.DataFrame:
        dati = dati.drop_duplicates()
        # Riassegna gli indici dopo l'eliminazione
        dati = dati.reset_index(drop=True)
        return dati


    #Metodo che tiene solo le features rilevanti:Clump Thickness,Uniformity of Cell Size
    #Uniformity of Cell Shape, Marginal Adhesion, Single Epithelial Cell Size
    #Bare Nuclei, Bland Chromatin, Normal Nucleoli, Mitoses
    def elimina_features(self,dati: pd.DataFrame) -> pd.DataFrame:
        features_eliminate=['Blood Pressure','Sample code number', 'Heart Rate']
        for feature in features_eliminate:
            dati= dati.drop(columns=[feature],axis=1)

        return dati

    #metodo eliminazione dei valori nulli (Nan e <null>)
    def elimina_nulli(self ,dati: pd.DataFrame) -> pd.DataFrame:
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

dati = data_csv('./dati/version_1.csv')
print(dati.data)

#decidi se moda media o mediana
#fai la factory
#se più di due null sul record eliminalo
#valuta se eliminare il record dove il class type è nullo




