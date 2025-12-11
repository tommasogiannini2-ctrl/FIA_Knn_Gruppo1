
from Prepocessing import *

class KNNClassifier:

    def __init__(self, db_features,db_classe):
        # Inizzializzo
        self.db_features = db_features
        self.db_classe = db_classe
        #inserisci la percentuale di training sed che si desidera
        self.x_training,self.y_training = self.training_set(0.9)
        self.x_test , self.y_test= self.test_set()
        self.k= self.knn_k_ottimale(self.db_features, self.db_classe)

    def training_set(self,percentuale_training_set: float):
        lunghezza_db = len(self.db_features)
        lunghezza_training = int(percentuale_training_set * lunghezza_db)

        X_train = self.db_features[:lunghezza_training].values
        Y_train = self.db_classe[:lunghezza_training].values
        # assegna i dati di addestramento
        self.x_training = X_train
        self.y_training = Y_train
        return self.x_training, self.y_training

    def test_set(self):
        lunghezza_test=len(self.db_features)-len(self.x_training)

        X_test = self.db_features[len(self.db_features)-lunghezza_test:].values
        Y_test =self.db_classe[len(self.db_classe)-lunghezza_test:].values
        self.x_test = X_test
        self.y_test = Y_test
        return self.x_test, self.y_test

    def distanza_euclidea(self, a, b) -> float:
        # calcola la distanza euclidea tra due punti
        # eventualmente aggiungere le altre due
        #a-> saranno le features di x_test
        #b-> saranno le features di x_training
        return np.sqrt(np.sum((a - b)**2))

    def classificazione(self, x_test) -> tuple[int,float,float]:
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

        classe_2 = conteggio_classi.get(2, 0)
        prob_predetta_2 = classe_2/ self.k
        classe_4=conteggio_classi.get(4, 0)
        prob_predetta_4 = classe_4/ self.k

        return classe_maggiore, prob_predetta_2, prob_predetta_4

    def knn_k_ottimale(self, db_features, db_classe)->int:

        lista_k = []
        lista_errori = []
        K_limite_superiore = len(self.x_training)
        # K_max per la ricerca: Usiamo 51 o K_limite_superiore, scegliendo il valore più piccolo
        K_max = min(51, K_limite_superiore-1)

        for k_attuale in range(3,K_max, 2):

            self.k = k_attuale
            errori = 0

            for x_campione, y_reale in zip(self.x_test,self.y_test):


                classe_predetta, prob_2, prob_4 = self.classificazione(x_campione)

                if classe_predetta != y_reale:
                    errori += 1

            tasso_errore_percent = errori / len(self.x_test)*100


            lista_k.append(k_attuale)
            lista_errori.append(tasso_errore_percent)

        min_errore_valore = min(lista_errori)

        indice_min_errore = lista_errori.index(min_errore_valore)


        k_best = lista_k[indice_min_errore]

        self.k = k_best  # Imposta il k ottimale trovato sul modello


        return k_best

    def  restituzione_classepredetta_probabilitaclasse4(self,):
        #questo metodo serve per il calcolo delle metriche, in quanto necessitano di questi due dati
        classe_predetta=[]
        prob_class_4=[]
        X_test = self.x_test
        for campione in X_test:
            _classe_predetta, prob_classe_2,_proba_class_4 = self.classificazione(campione)
        classe_predetta.append(_classe_predetta)
        prob_class_4.append(_proba_class_4)

        return classe_predetta, prob_class_4

nomefile = './dati/version_1.csv'

opener = scegli_opener(nomefile)

dati = DataCsv(opener,nomefile)
tupla = dati.load()
data_unico = unificaDF(tupla[0],tupla[1])
classificatore=KNNClassifier(tupla[0],tupla[1])

print(f"\n K OTTIMALE")
print(f"K Ottimale trovato dal modello: k={classificatore.k}")