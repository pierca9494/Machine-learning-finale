import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Genera 50 valori interi univoci casuali tra 1 e 100 (o altro range se necessario)
data = np.random.choice(np.arange(1, 101), size=50, replace=False)

# Step 2: Modifica la forma in una matrice 10x5
data_reshaped = data.reshape(10, 5)

# Step 3: Assicurati che ogni riga abbia solo valori interi e univoci
# Controlla che tutti i valori siano interi
if not np.all(np.mod(data_reshaped, 1) == 0):
    # Conversione esplicita a interi per garantire che siano tutti numeri interi
    data_reshaped = data_reshaped.astype(int)

# Controlla che ogni riga abbia valori univoci
for i in range(data_reshaped.shape[0]): #scorre per tutte le righe 
    #controlla  x ogni riga il numero dei valori univoci nelle colonne non sia minore di 5 
    while len(np.unique(data_reshaped[i])) < data_reshaped.shape[1]: 
        # Rigenera la riga se ci sono duplicati
        data_reshaped[i] = np.random.choice(np.arange(1, 101), size=5, replace=False)

# Step 4: Crea un DataFrame
df = pd.DataFrame(data_reshaped, columns=[f'Colonna_{i+1}' for i in range(5)])

# Step 5: Stampa il grafico
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.plot(df.columns, df.iloc[i], marker='o', label=f'Riga {i+1}')

plt.title("Visualizzazione del Dataset con Valori Univoci per Riga")
plt.xlabel("Colonne")
plt.ylabel("Valori")
plt.legend()
plt.grid()
plt.show()

# Stampa il DataFrame per controllare i valori
print(df)
