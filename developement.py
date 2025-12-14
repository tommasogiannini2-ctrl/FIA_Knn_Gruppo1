from Prepocessing import *
from validation_evaluation_strategies import *
import numpy as np

class KNNClassifier:

    def __init__(self, training, test):
        self.training = training
        self.test = test
        self.x_training = None
        self.y_training = None
        self.x_test = None
        self.y_test = None
        self.k = None

    def separatore(self, training, test):
        self.y_training = training['classtype_v1']
        self.x_training = training.drop(columns=['classtype_v1'])
        self.y_test = test['classtype_v1']
        self.x_test = test.drop(columns=['classtype_v1'])

    def distanza_euclidea(self, a, b) -> float:
        # calcola la distanza euclidea tra due punti
        # eventualmente aggiungere le altre due
        #a-> saranno le features di x_test
        #b-> saranno le features di x_training
        return np.sqrt(np.sum((a - b)**2))

    def classificazione(self, x) -> tuple[int,float,float]:
        distanze = []
        # per ogni campione dell'insieme di test, calcola la distanza da tutti i campioni del set di training
        for valori in self.x_training.values:
            euclidea = self.distanza_euclidea(x, valori)
            distanze.append(euclidea)

        # creazione DataFrame di distanze e classi
        col1 = pd.Series(distanze, name = 'distanze')
        col2 = pd.Series(self.y_training.values, name = 'y_training')
        tab = pd.concat([col1, col2], axis = 1) #Controllare se concat funziona (Gabri e Flavia)

        # estrazione le etichette dei K vicini
        tab_ordinata = tab.sort_values(by = ['distanze'], ascending = True)
        indici_k = tab_ordinata.head(self.k)

        # estrazione della classe più frequente
        colonna_classi = indici_k['y_training']
        conteggio_classi = colonna_classi.value_counts()
        classe_maggiore = conteggio_classi.index[0]

        classe_2 = conteggio_classi.get(2, 0)
        prob_predetta_2 = classe_2/ self.k
        classe_4=conteggio_classi.get(4, 0)
        prob_predetta_4 = classe_4/ self.k

        return classe_maggiore, prob_predetta_2, prob_predetta_4

    def knn_k_ottimale(self)->int:
        lista_k = []
        lista_errori = []
        K_limite_superiore = len(self.x_training)
        # K_max per la ricerca: Usiamo 51 o K_limite_superiore, scegliendo il valore più piccolo
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

        for k_attuale in range(3,K_max, 2):
            self.k = k_attuale
            errori = 0

            for x_campione, y_reale in zip(x_validation, y_validation):
                classe_predetta, prob_2, prob_4 = self.classificazione(x_campione)
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

    def  restituzione_classepredetta_probabilitaclasse4(self, x):
        #questo metodo serve per il calcolo delle metriche, in quanto necessitano di questi due dati
        classe_predetta=[]
        prob_class_4=[]

        for campione in x:
            _classe_predetta, prob_classe_2,_proba_class_4 = self.classificazione(campione)
            classe_predetta.append(_classe_predetta)
            prob_class_4.append(_proba_class_4)

        return classe_predetta, prob_class_4