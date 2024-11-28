import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

class MNISTClassifier:
    def __init__(self):
        self.digits = load_digits()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def explore_data(self):
        print("Informazioni sul dataset:")
        print(f"Shape delle immagini: {self.digits.images.shape}")
        print(f"Shape dei dati: {self.digits.data.shape}")
        print(f"Numero di classi: {len(np.unique(self.digits.target))}")
        print(f"Classi disponibili: {np.unique(self.digits.target)}")
        print(f"Dimensione di ogni immagine: {self.digits.images[0].shape}")
        print("\nPrimi 5 valori del target:")
        print(self.digits.target[:5])
        print("\nEsempio di dati (prima immagine in formato array):")
        print(self.digits.images[0])
        
        # Istogramma delle classi
        plt.figure(figsize=(8, 5))
        plt.hist(self.digits.target, bins=len(np.unique(self.digits.target)), rwidth=0.8, color='skyblue', edgecolor='black')
        plt.title("Distribuzione delle Classi")
        plt.xlabel("Classe")
        plt.ylabel("Frequenza")
        plt.show()

    def visualize_data(self):
        fig, axes = plt.subplots(1, 10, figsize=(10, 3))
        for i, ax in enumerate(axes):
            ax.imshow(self.digits.images[i], cmap='gray')
            ax.axis('off')
        plt.suptitle("Esempi di cifre MNIST")
        plt.show()

    def preprocess_data(self):
        X, y = self.digits.data, self.digits.target
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    def choose_model(self, kernel='linear'):
        self.model = SVC(kernel=kernel, random_state=42)

    def train_model(self):
        if self.model is None:
            raise ValueError("Model not initialized. Call choose_model() first.")
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuratezza del modello: {accuracy:.2f}")
        print("\nReport di Classificazione:\n")
        print(classification_report(self.y_test, y_pred))

        # Visualizzazione dei Risultati
        fig, axes = plt.subplots(1, 5, figsize=(10, 3))
        for i, ax in enumerate(axes):
            ax.imshow(self.X_test[i].reshape(8, 8), cmap='gray')
            ax.set_title(f"Pred: {y_pred[i]}\nTrue: {self.y_test[i]}")
            ax.axis('off')
        plt.suptitle("Predizioni e Valori Reali")
        plt.show()

        # Matrice di Confusione
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.digits.target_names, yticklabels=self.digits.target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Matrice di Confusione')
        plt.show()

# Utilizzo della classe
classifier = MNISTClassifier()
classifier.explore_data()
# classifier.visualize_data()
# classifier.preprocess_data()
# classifier.choose_model(kernel='linear')
# classifier.train_model()
# classifier.evaluate_model()
