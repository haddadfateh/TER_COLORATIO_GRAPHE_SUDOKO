
#best one
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# fonction convertire un nomre en un vecteur sur 4 bite
def one_hot_encode(number):
    """Convertir un nombre (0-4) en un vecteur one-hot de longueur 4."""
    one_hot = np.zeros(4, dtype=int)
    if number > 0:
        one_hot[number - 1] = 1
    return one_hot

# Préparer les données sous forme de tenseur 4x4x4 adapté au traitement CNN
def prepare_4_channel_grid(data_row):
    """transform dun  un tableau 1D de 16 cellules Sudoku en un tenseur  4x4x4."""
    grid = np.array(data_row).reshape(4, 4)
    channels = np.zeros((4, 4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            channels[:, i, j] = one_hot_encode(grid[i, j])
    return channels

# definition des modeles
class SudokuCNNPure(nn.Module):
    def __init__(self):
        super(SudokuCNNPure, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(32, 4, kernel_size=3, padding="same")  # Garder la même taille de sortie que les 3 premières couches

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)

        return F.softmax(x, dim=1)

# chargement des données
puzzles_4x4 = pd.read_csv('sudoku_puzzles_4x4.csv')
solutions_4x4 = pd.read_csv('sudoku_solutions_4x4.csv')

# Préparer les ensembles de données
X = np.stack([prepare_4_channel_grid(row) for row in puzzles_4x4.to_numpy()])
Y = np.stack([prepare_4_channel_grid(row) for row in solutions_4x4.to_numpy()])

Y = np.argmax(Y, axis=1)  # indice de class



# Convertir en tenseurs
X_train = torch.tensor(X[:800], dtype=torch.float32)
Y_train = torch.tensor(Y[:800], dtype=torch.long)
X_val = torch.tensor(X[800:], dtype=torch.float32)
Y_val = torch.tensor(Y[800:], dtype=torch.long)




Y_train = torch.transpose(Y_train, 2, 1)



Y_val = torch.transpose(Y_val, 2, 1)

print(X_val.shape)
print(Y_val.shape)

# initialise le modl , la fonction de perte et l'optimiseur
model = SudokuCNNPure()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# fction pour entrainer le modele et evaluer sur lensemble de validation
def train_and_evaluate(model, criterion, optimizer, X_train, Y_train, X_val, Y_val, epochs=500):
    model.train()
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(epochs), desc="Entraînement en cours"):
        optimizer.zero_grad()
        outputs_train = model(X_train)

        outputs_train = torch.transpose(outputs_train, 3, 1)

        outputs_train = outputs_train.reshape(800*4*4,4)

        Y_train = Y_train.reshape(800*4*4)


        loss_train = criterion(outputs_train, Y_train)




        loss_train.backward()#Calcule le gradient de la perte par rapport aux paramètres du modèle
        optimizer.step()
        train_losses.append(loss_train.item())

        # parite ou on ft levaluation
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val)
            loss_val = criterion(outputs_val, Y_val)
            val_losses.append(loss_val.item())

        if epoch % 50 == 0:
            print(f'Epoch {epoch + 1}: Loss Entraînement: {loss_train.item():.4f}, Loss Validation: {loss_val.item():.4f}')

    return train_losses, val_losses

train_losses, val_losses = train_and_evaluate(model, criterion, optimizer, X_train, Y_train, X_val, Y_val, epochs=500)

# TRACER LA PERSTE dentrain
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Loss Entraînement')
plt.plot(val_losses, label='Loss Validation')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.title('Perte d\'entraînement et de validation')
plt.legend()
plt.show()

#EVALUER LE MODELE
def evaluate_model(model, X_val, Y_val):
    model.eval()
    with torch.no_grad():

        outputs = model(X_val)

        _, predicted_classes = torch.max(outputs, 1)

        print(predicted_classes[0])
        correct = (predicted_classes == Y_val).float()
        accuracy = correct.sum() / correct.numel()
        print(f'Précision de validation: {accuracy.item() * 100:.2f}%')

evaluate_model(model, X_val, Y_val)

