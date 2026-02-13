from Prepocessing import *
from development import *
from validation_evaluation_strategies import *
import argparse
import numpy as np
import pandas as pd
import os

parser = argparse.ArgumentParser(
    description='Elabora un dataframe secondo il metodo KKN e calcola le metriche più comuni.')

default_input = os.getenv('IN_DIR', './dati')
default_output = os.getenv('OUT_DIR', './risultati')

# definisce l'argomento per il file di ingresso e di uscita con un valore di default
parser.add_argument('-i', '--input', type=str, default=os.path.join(default_input, 'version_1.csv'),
                    help='Inserire percorso del file di ingresso (Default: dati/version_1.csv)')
parser.add_argument('-o', '--output', type=str, default=default_output,
                    help='Inserire percorso della cartella di uscita (Default: risultati)')
parser.add_argument('-v', '--validation', type=str, default=None, required=True, choices=['RS', 'KF'],
                    help='Scegliere il metodo di validazione da eseguire (Inserire RS per eseguire il Random Subsampling o KF per eseguire il K-Fold Cross Validation)')
parser.add_argument('-p', '--percentuale_holdout', type=float, default=0.8,
                    help="Scegliere percentuale per l'holdout (Default: 0.8)")
parser.add_argument('-K', '--K_prove', type=int, default=5,
                    help='Scegliere il numero di esperimenti da eseguire per il Random Subsampling o per il K-Fold Cross Validation (Default=5)')
pars = parser.parse_args()

pars_out = pars.output
validation_type = pars.validation
p_Holdout = pars.percentuale_holdout
n_prove = pars.K_prove

# Scelta opener a seconda dell'estensione del file di ingresso
filename = pars.input
opener = scegli_opener(filename)

# Creaione cartelle per i file di uscita (se non esiste già)
if pars_out:
    cartella_gia_esistente = os.path.exists(pars_out)
    os.makedirs(pars_out, exist_ok=True)
    if not cartella_gia_esistente:
        print(f"Cartella {pars_out} creata \n")

dati = Data(opener,filename)
tupla = dati.load()

# Dataframe unico e pulito, da questo bisognerà dividere in training e set
data_unico = unificaDF(tupla[0],tupla[1])

# factory per  Holdout
factory_holdout = validation_factory('Holdout', data_unico)
lista_holdout = factory_holdout.split(n=1, p=p_Holdout)


factory_main = validation_factory(validation_type, data_unico)

lista = factory_main.split(k_prove=n_prove, n=n_prove, p=0.8)

# Calcolo metriche per Holdout
coppia_holdout = lista_holdout[0]
training_holdout = coppia_holdout[0]
test_holdout = coppia_holdout[1]
print("Stuttura dei datframe di training e test per esperimento Holdout. \n")
print(training_holdout.info())
print(test_holdout.info())
risultati_Holdout, evaluation_holdout = calcolo_metriche(training_holdout, test_holdout)
print("Ottenuti risultati per esperimento di Holdout. \n")

# Grafici Holdout
plotter = Plot()
plotter.plot_matrice_confusione(evaluation_holdout.confusion_matrix, pars_out)
plotter.plot_roc_curve(evaluation_holdout.FPR, evaluation_holdout.TPR, risultati_Holdout['auc'],pars_out)

# Calcolo metriche per ogni esperimento RS o KF
risultati = []
for i in range(n_prove):
    coppia = lista[i]
    training = coppia[0]
    test = coppia[1]
    print(f"Stuttura dei datframe di training e test per esperimento {i+1} di {validation_type}. \n")
    print(training.info())
    print(test.info())
    ris, e = calcolo_metriche(training, test)
    risultati.append(ris)
    print(f"Ottenuti risultati per esperimento {i+1} di {validation_type} \n")

# Calcolo medie e deviazioni standard delle metriche
risultati_finali = calcolo_media_stddev_metriche(risultati)
r_tot = unisci_risultati(risultati_Holdout, risultati, risultati_finali)
# Output in un file Excel
path_excel=pars_out +"/risultati.xlsx"
r_tot.to_excel(path_excel, index=False)

# Grafico RS/KF
plotter.plot_distribuzione_performance(pd.DataFrame(risultati),pars_out)

print(f"I risultati vengono salvati in {path_excel} \n")
