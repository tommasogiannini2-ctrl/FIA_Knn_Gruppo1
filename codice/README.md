#  Progetto FIA - Classificatore di Tumori k-NN (Gruppo 1): Giannini Tommaso, di Raimondo Gabriele, Romito Flavia



## Obiettivo del Progetto

Il progetto mira a sviluppare una \*\*pipeline di Machine Learning\*\* generica in Python che addestra e valuta un classificatore \*\*k-Nearest Neighbors (k-NN)\*\* . Il modello è utilizzato per classificare le cellule tumorali come benigne (Classe 2) o maligne (Classe 4) in base a 9 caratteristiche fornite.

---



## Configurazione e Specifiche Tecniche



### Parametri Definitivi del Gruppo (i=1)

| \*\*A (Input Data)\*\* | `version\_1.csv` | I dati di input devono essere caricati da un file in formato \*\*CSV\*\*. |

| \*\*B (Validazione 1)\*\* | \*\*`Random Subsampling`\*\* | Tecnica di valutazione richiesta in modalità \*K\* esperimenti. |

| \*\*C (Validazione 2)\*\* | \*\*`K-fold Cross Validation`\*\* | Seconda tecnica di valutazione richiesta in modalità \*K\* esperimenti (deve gestire le folds). |



### Modularizzazione e Pattern

* Il codice include la \*\*verifica automatica di almeno 3 test di correttezza\*\* tramite il modulo `unittest`.
* Il codice include l'implementazione del pattern Factory per scegliere in maniera ottimale l'opener da utilizzare per aprire il file dei dati in ingresso.



### Dipendenze

Sono consentiti e utilizzati i seguenti pacchetti non standard:

* `numpy` (per operazioni vettoriali e matriciali).

* `pandas` (per il caricamento e la manipolazione del dataset).

* `matplotlib` (per la visualizzazione dei risultati, es. Confusion Matrix, ROC).

*'ABC' per l'implemetazione delle factory.

*'openpyxl' per il salvataggio dei risultati in un file excel.

---


## Dataset e Data Preprocessing
I dati iniziali letti da un file di input grazie ad un opener scelto con il pattern Factory.
Successivamenti i dati sono processati secondo i seguenti passaggi:
* Eliminazione record duplicati.
* Sostituzione con il valore NaN delle feature con valori fuori dal range [1, 10].
* Sostituzione con il valore NaN dei valori diversi da 2 o 4 nella classe obiettivo.
* Eliminazione record con classe obiettivo nulla.
* Divisione del dataset in features e classe obiettivo.
* Eliminazione delle features non rilevanti (non presenti nella specifica).
* Eliminazione record con più di 4 features nulle.
* Sostituzione dei valori NaN nei record rimanenti con la moda della feature corrispondente.

Vengono quindi restituiti il dataframe delle feature e la colonna della classe obiettivo.


## Istruzioni per l'Esecuzione


1\. Clonare il repository GitHub (su una macchina che supporta Docker e Docker Compose):

&nbsp; git clone https://github.com/tommasogiannini2-ctrl/FIA_Knn_Gruppo1.git

2\. Entrare nella directory FIA_Knn_Gruppo1

&nbsp; cd ./FIA_Knn_Grruppo1

3\. Eseguire il comando per avere l'Help del programma per capire le opzioni di esecuzione possibili, attivando anche il Docker e il Docker Compose

&nbsp; docker-compose run app-knn python main.py -h

4\. Eseguire il codice con le opzioni desiderate.
Esempio per avere una esecuzione con il metodo KFold e un numero di prove pari a 3

&nbsp; docker-compose run app-knn python main.py -v KF -K 3


### Guida all'esecuzione del programma

Il programma contiene 6 parser, **il solo obbligatorio è '-v'** che consente di scegliere se effettuare un Random Subsampling (inserire RS) o la K-Fold Cross Validation (inserire KF).
Gli altri parser sono:
* -i, --input: richiede una striga che specifica il path del file di ingresso (Default: ./dati/version_1.csv)
* -o, --output: richiede una stringa che specifica il pth del file di uscita (Default: risultati.xlsx)
* -v, --validation: richiede di inserire RS o KF
* -K, --K_prove: richiede di inserire il numero dei K esperimenti
* -p, --percentuale_holdout: richiede di inserire la percentuale per effettuale l'Holdout
* -h, --help: consente di visualizzare tutti i comandi precedenti

Una volta eseguito il main con le impostazioni desiderate, l'esecuzione mostrerà:
* Un riepilogo delle colonne del dataframe dopo il processo di pulizia, e il numero di elementi (nulli e non nulli) che esso contiene.
* Un riepilogo della classe obiettivo dopo il processo di pulizia, e il numero di elementi (nulli e non nulli) che essa contiene.
* Un riepilogo della struttura dei dataframe di training e test per l'esperimento di Holdout e una conferma che i risultati siano stati calcolati.
* Un riepilogo della struttura dei dataframe di training e test per ogni esperimenti di KFold o Random Subsampling e una conferma che i risultati siano stati calcolati.
* Infine una conferma che i risultati siano stati salvati nella cartella scelta precedentemente.

---

## Risultati

I risultati vengono salvati in maniera automatica in una cartella denominata "risultati".
Grazie al parser -o --output è possibile indicare un altro path per il salvataggio.

In questa cartella saranno presenti:
* Un file excel contenente le metriche di tutti gli esperimenti, ovvero una colonna per
l'esperimento di Holdout e K colonne per gli esperimenti di KFold o Random Subsampling,
con media e deviazione standard relative a quest'ultimi.
* Un grafico contenente la Matrice di Confusione relativa all'esperimento di Holdout.
* Un grafico contenente la curva ROC relativa all'esperimento di Holdout.
* Un grafico (BoxPlot) contenente la distribuzione delle metriche relative agli esperimenti di KFold o Random Subsampling.


