import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Importazione dei Dati
digits = load_digits()
X, y = digits.data, digits.target

# Visualizzazione di alcune cifre
fig, axes = plt.subplots(1, 10, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap='gray')
    ax.axis('off')
plt.suptitle("Esempi di cifre MNIST")
plt.show()

# Preprocessing dei Dati
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Divisione in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Scelta del Modello
model = SVC(kernel='linear', random_state=42)

# Addestramento del Modello
model.fit(X_train, y_train)

# Valutazione del Modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello: {accuracy:.2f}")
print("\nReport di Classificazione:\n")
print(classification_report(y_test, y_pred))

# Visualizzazione dei Risultati
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
    ax.axis('off')
plt.suptitle("Predizioni e Valori Reali")
plt.show()

# Matrice di Confusione
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Matrice di Confusione')
plt.show()

# Esperimenti Extra: Cross-validation e Algoritmo Alternativo
# Commentato per esecuzione rapida, sbloccare per esperimenti
# from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier
#
# knn_model = KNeighborsClassifier(n_neighbors=3)
# scores = cross_val_score(knn_model, X_scaled, y, cv=5)
# print(f"Accuratezza media con k-NN: {np.mean(scores):.2f}")
