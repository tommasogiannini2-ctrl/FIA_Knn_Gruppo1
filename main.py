from Prepocessing import *
from developement import *
from validation_evaluation_strategies import *
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Elabora un dataframe secondo il metodo KKN e calcola le metriche più comuni.')

# definisce l'argomento per il file di ingresso e di uscita con un valore di default
parser.add_argument('-i', '--input', type=str, default='dati/version_1.csv', help='Inserire percorso del file di ingresso (Default: dati/version_1.csv)')
parser.add_argument('-o', '--output', type=str, default='risultati.xlsx',help='Inserire percorso del file JSON di uscita (Default: risultati.xlsx)')
parser.add_argument('-v', '--validation', type=str, default=None, required=True, choices=['RS','KF'], help='Scegliere il metodo di validazione da eseguire (Inserire RS per eseguire il Random Subsampling o KF per eseguire il K-Fold Cross Validation)')
parser.add_argument('-p', '--percentuale_holdout', type=float, default=0.8, help="Scegliere percentuale per l'holdout (Default: 0.8)")
parser.add_argument('-K', '--K_prove', type=int, default=5, help='Scegliere il numero di esperimenti da eseguire per il Randoma Subsampling o per il K-Fold Cross Validation (Default=5')
pars = parser.parse_args()

pars_out = pars.output
validation_type = pars.validation
p_Holdout = pars.percentuale_holdout
n_prove = pars.K_prove

filename = pars.input
opener = scegli_opener(filename)

dati = Data(opener,filename)
tupla = dati.load()

# Dataframe unico e pulito, da questo bisognerà dividere in training e set
data_unico = unificaDF(tupla[0],tupla[1])

# Parametri per la divisione
data = ValidationStrategy(data_unico)
p_RandomSubsampling = 0.8

lista = []
if validation_type == 'RS':
    lista = data.RandomSubsampling(n_prove, p_RandomSubsampling)
elif validation_type == 'KF':
    lista = data.Kfold(n_prove)

# Divisione con holdout
lista_holdout = data.RandomSubsampling(1,p_Holdout)
# Questa lista contiene una coppia di training e test divise secondo il metodo Holduot

"""
Controllo sulle dimensioni
coppia = lista_holdout[0]
print('HOLDOUT---------------')
print(coppia[0].info())
print(coppia[1].info())
"""

"""
# Divisione training e set con il metodo Kfold
lista_Kfold = data.Kfold(n_prove)
# Questa lista contiene n coppie di training e test divise secondo il metodo Kfold

#Ciclo per vedere se funziona tutto
print('K FOLD---------------')

for i in range(n_prove):
    coppia = lista_Kfold[i]
    print(f"coppia {i + 1}")
    print(coppia[0].info())
    print(coppia[1].info())

# Divisione training e set con il metodo RandomSubsampling
lista_RS = data.RandomSubsampling(n_prove, p_RandomSubsampling)
# Questa lista contiene n coppie di training e test divise secondo il metodo Random Subsampling

#Ciclo per vedere se funziona tutto
print('RANDOM SUBSAMPLING---------------')

for i in range(n_prove):
    coppia = lista_RS[i]
    print(f"coppia {i + 1}")
    print(coppia[0].info())
    print(coppia[1].info())
"""

training_holdout = lista_holdout[0]
test_holdout = lista_holdout[1]
classificatoreH = KNNClassifier(training_holdout, test_holdout)
classificatoreH.separatore()
k_ott_holdout = classificatoreH.knn_k_ottimale()
y_predette, prob_class_4 = classificatoreH.restituzione_classepredetta(classificatoreH.x_test.values)
classe_vera = np.round(classificatoreH.y_test.values).astype(int)
classe_predetta = np.round(y_predette).astype(int)

evaluation = Evaluation(classe_vera, classe_predetta)
evaluation.matrice_confusione()

thresholds = np.linspace(1.0, 0.0, 1001)
auc = evaluation.area_under_the_curve(classificatoreH.y_test.values, thresholds, prob_class_4)

risultatiH = {
        'k': k_ott_holdout,
        'accuracy': evaluation.accuracy_rate(),
        'error_rate': evaluation.error_rate(),
        'sensitivity': evaluation.sensitivity(),
        'specificity': evaluation.specificity(),
        'geometric_mean': evaluation.geometric_mean(),
        'auc': auc
    }

#Una volta ottenuto training e set si trova la k ottima sul training

#Trovata la k ottima si esegue sul test e si calcolano le metriche

risultati = []

for training, test in lista:

    classificatore = KNNClassifier(training, test)
    classificatore.separatore()

    k_ottimale = classificatore.knn_k_ottimale()
    y_predette, prob_class_4 = classificatore.restituzione_classepredetta(classificatore.x_test.values)

    classe_vera = np.round(classificatore.y_test.values).astype(int)
    classe_predetta = np.round(y_predette).astype(int)

    evaluation = Evaluation(classe_vera, classe_predetta)
    evaluation.matrice_confusione()

    thresholds = np.linspace(1.0, 0.0, 1001)
    auc = evaluation.area_under_the_curve(classificatore.y_test.values, thresholds, prob_class_4)

    # Aggiungi risultati
    risultati.append({
        'k': k_ottimale,
        'accuracy': evaluation.accuracy_rate(),
        'error_rate': evaluation.error_rate(),
        'sensitivity': evaluation.sensitivity(),
        'specificity': evaluation.specificity(),
        'geometric_mean': evaluation.geometric_mean(),
        'auc': auc
    })

risultati = pd.DataFrame(risultati)
metriche = risultati.columns.drop('k')

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

k_medio = risultati['k'].mean()

risultati_finali = pd.DataFrame({
    'Metrica': lista_metrica,
    'Media': lista_media,
    'Deviazione Standard': lista_devstd
})

risultati_finali.loc[-1] = ['K Ottimale Medio', k_medio, np.nan]
risultati_finali.index = risultati_finali.index + 1
risultati_finali = risultati_finali.sort_index().reset_index(drop=True)

risultati_finali.to_excel(pars.output, index=False)

#Parte che non so se serve
#classificatore=KNNClassifier(tupla[0],tupla[1])

#print(f"\n K OTTIMALE")
#print(f"K Ottimale trovato dal modello: k={classificatore.k}")

#input("Premi INVIO per chiudere...")
