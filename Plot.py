import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self):
        pass

    def plot_matrice_confusione(self, cm):
        nomi_classi = ['Negativo (2)', 'Positivo (4)']

        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matrice di Confusione HOLDOUT')
        plt.colorbar()

        tick_marks = np.arange(len(nomi_classi))
        plt.xticks(tick_marks, nomi_classi)
        plt.yticks(tick_marks, nomi_classi)

        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Classe Vera')
        plt.xlabel('Classe Predetta')
        plt.tight_layout()
        plt.savefig('risultati/confusion_matrix_holdout.png')

    def plot_roc_curve(self, FPR, TPR, auc):
        plt.figure(figsize=(7, 6))
        plt.plot(FPR, TPR, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve HOLDOUT')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig('risultati/ROC_curve.png')

    def plot_distribuzione_performance(self, risultati):
        metriche = risultati.drop(columns=['k'])
        plt.figure(figsize=(10, 6))
        metriche.boxplot()
        plt.title('Distribuzione delle Performance sulle Folds')
        plt.ylabel('Valore Metrica')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('risultati/distribuzione_performance.png')
