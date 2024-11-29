import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#creazione della classe del nostro dataset di riferimento
class IrisDataset:
    def __init__(self):
        self.data = load_iris()
        self.X = self.data.data
        self.y = self.data.target
        self.target_names = self.data.target_names
        self.feature_names = self.data.feature_names

    def esplora_dati(self):
        print("Caratteristiche del dataset:")
        print(self.feature_names)
        print("\nClassi target:")
        print(self.target_names)
        print("\nPrime 5 righe dei dati:")
        print(pd.DataFrame(self.X, columns=self.feature_names).head())
#creazione classi modello 
class IrisModel:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_preprocessed = False # implementazione gestione errori 
        self.is_trained = False #dato che nella prima versione nell'app se si valutava il modello prima di addestrarlo dava errore 

    def preprocess_data(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataset.X, self.dataset.y, test_size=test_size, random_state=random_state
        )
        self.is_preprocessed = True

    def train_model(self):
        if not self.is_preprocessed: #gestione errore nell'app
            raise ValueError("Errore: i dati devono essere preprocessati prima di addestrare il modello.")
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True

    def evaluate_model(self):
        if not self.is_trained:
            raise ValueError("Errore: il modello deve essere addestrato prima di poter essere valutato.")
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuratezza: {accuracy:.2f}")
        print("\nReport di classificazione:")
        print(classification_report(self.y_test, y_pred, target_names=self.dataset.target_names))
        return y_pred

    def plot_confusion_matrix(self, y_pred):
        if not self.is_trained:
            raise ValueError("Errore: il modello deve essere addestrato prima di poter visualizzare la matrice di confusione.")
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", 
                    xticklabels=self.dataset.target_names, 
                    yticklabels=self.dataset.target_names)
        plt.title("Matrice di Confusione")
        plt.xlabel("Predizione")
        plt.ylabel("Reale")
        plt.show()

class IrisApp:
    def __init__(self):
        self.dataset = IrisDataset()
        self.model = IrisModel(self.dataset)

    def run(self):
        while True:
            print("\nMenu Applicazione Iris:")
            print("1. Esplora il dataset")
            print("2. Preprocessa i dati")
            print("3. Addestra il modello")
            print("4. Valuta il modello")
            print("5. Visualizza matrice di confusione")
            print("6. Esci")

            scelta = input("Scegli un'opzione: ")

            try:
                if scelta == "1":
                    self.dataset.esplora_dati()

                elif scelta == "2":
                    self.model.preprocess_data()
                    print("Dati preprocessati con successo.")

                elif scelta == "3":
                    self.model.train_model()
                    print("Modello addestrato con successo.")

                elif scelta == "4":
                    y_pred = self.model.evaluate_model()

                elif scelta == "5":
                    y_pred = self.model.evaluate_model()
                    self.model.plot_confusion_matrix(y_pred)

                elif scelta == "6":
                    print("Uscita dall'applicazione.")
                    break

                else:
                    print("Opzione non valida. Riprova.")

            except ValueError as e:
                print(e)

if __name__ == "__main__":
    app = IrisApp()
    app.run()
