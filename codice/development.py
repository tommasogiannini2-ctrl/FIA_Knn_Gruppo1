from validation_evaluation_strategies import *
import numpy as np
import pandas as pd
from Plot import *

class KNNClassifier:
    """
    Classe in cui è sviluppato l'algoritmo KNN e il calcolo del k ottimale
    Prende in ingresso due dataframe, uno di training e uno di test
    """

    def __init__(self, training, test):
        self.training = training
        self.test = test
        self.x_training = None
        self.y_training = None
        self.x_test = None
        self.y_test = None
        self.k = None

    def separatore(self):
        """
        Separa in feature e classe obiettivo entrambi i dataframe passati in ingresso alla classe
        """
        self.y_training = self.training['classtype_v1']
        self.x_training = self.training.drop(columns=['classtype_v1'])
        self.y_test = self.test['classtype_v1']
        self.x_test = self.test.drop(columns=['classtype_v1'])

    def distanza_euclidea(self, a, b) -> float:
        # calcola la distanza euclidea tra due punti
        return np.sqrt(np.sum((a - b)**2))

    def classificazione(self, x) -> tuple[int,float]:
        """
        Prende in ingresso un record x e lo classifica utilizzando l'algoritmo KNN
        Ritorna una tupla contenente la classe predetta per il record x e la probabilità che
        la classe predetta sia 4 (da cui si può ricavare la probabilità della classe 2)
        """

        #Per ogni campione dell'insieme di training, calcola la distanza del record x
        #dal record della classe di training
        #Ritorna un vettore delle distanze non ordinato
        distanze = []
        for valori in self.x_training.values:
            euclidea = self.distanza_euclidea(x, valori)
            distanze.append(euclidea)

        # creazione DataFrame di distanze e classi
        col1 = pd.Series(distanze, name = 'distanze')
        col2 = pd.Series(self.y_training.values, name = 'y_training')
        tab = pd.concat([col1, col2], axis = 1)

        # ordina le distanze in maniera decrescente ed estrae le etichette dei K vicini
        tab_ordinata = tab.sort_values(by = ['distanze'], ascending = True)
        indici_k = tab_ordinata.iloc[:self.k]

        # estrazione della classe più frequente
        colonna_classi = indici_k['y_training']
        conteggio_classi = colonna_classi.value_counts()
        classe_maggiore = conteggio_classi.index[0]

        classe_4=conteggio_classi.get(4, 0)
        prob_predetta_4 = classe_4/ self.k

        return classe_maggiore, prob_predetta_4

    def knn_k_ottimale(self)->int:
        """
        Questa funzione serve a ottimizzare l'iperparametro k per svolgere l'algoritmo KNN
        Ritorna il k ottimale facendo delle prove con vari valori di k e sceglie quello che
        commette il numero di errori minimo (se ci sono più k con un numero minimo di errori
        sceglie casualmente)
        """
        lista_k = []
        lista_errori = []
        K_limite_superiore = len(self.x_training)
        # K_max per la ricerca: Usiamo 51 o K_limite_superiore per non eseguire troppe prove
        # e non renderlo troppo pesante computazionalmente e evitare overfitting
        # scegliendo il valore più piccolo
        K_max = min(51, K_limite_superiore-1)

        # copia per ripristinarlo dopo
        x_training = self.x_training.copy()
        y_training = self.y_training.copy()

        # crea il set di validazione
        validation_strategy = ValidationStrategy(self.training)
        validation_set = validation_strategy.RandomSubsampling(1, 0.8)[0]
        training_k = validation_set[0]
        validation_k = validation_set[1]

        self.x_training = training_k.drop(columns=['classtype_v1'])
        self.y_training = training_k['classtype_v1']

        x_validation = validation_k.drop(columns=['classtype_v1']).values
        y_validation = validation_k['classtype_v1'].values


        """
        ciclo che per ogni valore di k dispari esegue l'algoritmo KNN su un 
        validationn set ottenuto suddividendo il training set in un nuovo training set
        e validation set (con metodo Holdout e probabilità Holdout 80%) 
        Salva gli errori che ogni k commette confrontando
        le predizioni con il valore reale della classe obiettivo
        Sceglie il k che ha errori minori
        Ritorna k ottimo
        """
        for k_attuale in range(3,K_max, 2):
            self.k = k_attuale
            errori = 0

            for x_campione, y_reale in zip(x_validation, y_validation):
                classe_predetta, prob_4 = self.classificazione(x_campione)
                if classe_predetta != y_reale:
                    errori += 1
            tasso_errore_percent = errori / len(validation_k)*100
            lista_k.append(k_attuale)
            lista_errori.append(tasso_errore_percent)

        min_errore_valore = min(lista_errori)
        indice_min_errore = lista_errori.index(min_errore_valore)
        k_best = lista_k[indice_min_errore]

        self.x_training = x_training
        self.y_training = y_training

        self.k = k_best  # Imposta il k ottimale trovato sul modello
        return k_best

    def  restituzione_classepredetta(self, x):
        """
        Questo metodo restituisce una lista di classe predette e di probabilità che la classe predetta sia 4,
        per ogni campione di un generico dataset
        Prende in ingresso un dataset di feature x
        Ritona una tupla di due liste la prima contiene tutte le predizioni per i record,
        la seconda la probabilità che la classe predetta sia 4
        """
        classe_predetta=[]
        prob_class_4=[]

        for campione in x:
            _classe_predetta, _proba_class_4 = self.classificazione(campione)
            classe_predetta.append(_classe_predetta)
            prob_class_4.append(_proba_class_4)

        return classe_predetta, prob_class_4


def calcolo_metriche(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame)-> tuple[dict, Evaluation] | None:
    """
    Questo metodo prende in ingresso un dataframe di test e uno di training, e calcola le
    metriche restituendole in un dizionario.
    Le metriche sono l'accuracy, l'error rate, sensitivity, specificity, geometric mean e auc.
    Viene inoltre restituito il k ottimo relativo al classificatore.
    """
    classificatore = KNNClassifier(dataframe1, dataframe2)
    classificatore.separatore()
    k_ottimale = classificatore.knn_k_ottimale()
    y_predette, prob_class_4 = classificatore.restituzione_classepredetta(classificatore.x_test.values)
    classe_vera = np.round(classificatore.y_test.values).astype(int)
    classe_predetta = np.array(y_predette).astype(int)

    evaluation = Evaluation(classe_vera, classe_predetta)
    evaluation.matrice_confusione()

    thresholds = np.linspace(1.0, 0.0, 1001)
    evaluation.FPR, evaluation.TPR = evaluation.roc_curve(classificatore.y_test.values, thresholds, prob_class_4)
    auc = evaluation.area_under_the_curve(evaluation.FPR, evaluation.TPR)

    risultati = {
        'k': k_ottimale,
        'accuracy': evaluation.accuracy_rate(),
        'error_rate': evaluation.error_rate(),
        'sensitivity': evaluation.sensitivity(),
        'specificity': evaluation.specificity(),
        'geometric_mean': evaluation.geometric_mean(),
        'auc': auc
    }
    return risultati, evaluation

def calcolo_media_stddev_metriche(lista_ris: list)-> pd.DataFrame | None:
    """
    Questo metodo serve a calcolare media e deviazione standard delle metriche relative
    ai k esperimenti effettuati con Kfold o Random Subsampling
    Prende in ingresso la lista dei risultati di tutti gli esperimenti
    Ritorna un dataframe contenete le colonne Metriche, Media e Deviazione Standard
    """
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

    k_mode = risultati['k'].mode()[0]

    risultati_finali = pd.DataFrame({
        'Metrica': lista_metrica,
        'Media': lista_media,
        'Deviazione Standard': lista_devstd
    })

    risultati_finali.loc[-1] = ['k', k_mode, np.nan]
    risultati_finali.index = risultati_finali.index + 1
    risultati_finali = risultati_finali.sort_index().reset_index(drop=True)

    return risultati_finali


def unisci_risultati(lista_1:dict,lista_2:list, dataframe:pd.DataFrame)-> pd.DataFrame:
    """
    Questa funzione unisce i risultati di tutti i test in un unico dataframe.
    :param lista_1: Risultati dell'esperimento Holdout
    :param lista_2: Risultati degli esperimenti con Kfold o Random Subsampling
    :param dataframe: Media e Deviazione standard degli esperimenti Kfold o Random Subsampling
    :return: un dataframe che contiene tutte queste informazioni in modo ordinato
    """
    #lista 1 è la lista dei risultati di holdout
    colonna_H=pd.Series(lista_1,name='Metriche Holdout')
    nuove_colonne=[]
    for i in range(len(lista_2)):
        nuove_colonne.append(f"Metriche esperimento {i+1}")
    dataframe_n = pd.DataFrame(lista_2)
    dataframe_nT=dataframe_n.T
    dataframe_nT.columns=nuove_colonne

    data_tot=pd.concat([colonna_H,dataframe_nT],axis=1)
    df_aggiuntivo_allineato = dataframe.set_index('Metrica')
    df_aggiuntivo_allineato.columns = ['Media', 'Deviazione Standard']
    data_finale = pd.concat([data_tot, df_aggiuntivo_allineato], axis=1)
    data_finale = data_finale.reset_index().rename(columns={'index': 'Metrica'})
    return data_finale




