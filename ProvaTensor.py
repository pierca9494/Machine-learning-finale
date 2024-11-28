import tensorflow as tf
from tensorflow import layers, models
from tensorflow import mnist
import matplotlib.pyplot as plt

# Carica il dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizza i valori dei pixel tra 0 e 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Aggiungi una dimensione per i canali (richiesto per i modelli convoluzionali)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Costruisci il modello
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compila il modello
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Addestra il modello
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Valutazione sul test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nAccuracy sul test set: {test_acc:.2f}")

# Visualizza alcune predizioni
predictions = model.predict(x_test)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap=plt.cm.binary)
    plt.title(f"Pred: {predictions[i].argmax()}")
    plt.axis('off')
plt.show()
