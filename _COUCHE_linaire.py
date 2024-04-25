
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

def one_hot_encode(number):
    """convertire un nombre avec on hote encoding sur 4 bit ."""
    one_hot = np.zeros(4, dtype=int)
    if number > 0:
        one_hot[number - 1] = 1
    return one_hot

def prepare_one_hot_vector_4x4(data_row):
    """convertire une ligne en un vecteur on hot 16*4."""
    grid = np.array(data_row).reshape(16)  # je flat la grille
    one_hot_grid = np.array([one_hot_encode(number) for number in grid])
    return one_hot_grid.flatten()

def decode_sudoku_grid(one_hot_encoded):
    """Traduit un grille de sudoku encodée en one-hot back à sa représentation numérique 4x4 avec les zéros intacts"""
    #pour la recuperattion si 0000 0 sino selon lindice
    grid = []
    for i in range(0, len(one_hot_encoded), 4):
        block = one_hot_encoded[i:i+4]
        if block.sum() == 0:  # No number is encoded
            grid.append(0)
        else:
            grid.append(np.argmax(block) + 1)  # Convert index to Sudoku number
    return np.array(grid).reshape(4, 4)

#je charge les puzel et les solution
puzzles_4x4 = pd.read_csv('sudoku_puzzles_4x4.csv')
solutions_4x4 = pd.read_csv('sudoku_solutions_4x4.csv')

# je prepar X et Y
X = np.array([prepare_one_hot_vector_4x4(row) for row in puzzles_4x4.to_numpy()])
Y = np.array([prepare_one_hot_vector_4x4(row) for row in solutions_4x4.to_numpy()])
Y = np.array([row.reshape(-1, 4).argmax(axis=1) for row in Y])  # je convertie  one-hot en   class dindices

#division des données dans lordre
train_size = 800
X_train = torch.tensor(X[:train_size], dtype=torch.float32)
Y_train = torch.tensor(Y[:train_size], dtype=torch.long)
X_val = torch.tensor(X[train_size:], dtype=torch.float32)
Y_val = torch.tensor(Y[train_size:], dtype=torch.long)

class SudokuNN(nn.Module):
    def __init__(self):
        super(SudokuNN, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x).view(-1, 16, 4)
        return F.log_softmax(x, dim=2)

model = SudokuNN()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, criterion, optimizer, X_train, Y_train, epochs=1000):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        optimizer.zero_grad()

        outputs = model(X_train)


        outputs = outputs.view(-1,4)



        Y_train = Y_train.view(-1)



        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.view(-1, 4), Y_val.view(-1))
            val_losses.append(val_loss.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch + 1}: Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')
        model.train()

    # jafiche le plot des loss pour validation et train model
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return train_losses, val_losses

train_losses, val_losses = train_model(model, criterion, optimizer, X_train, Y_train, epochs=1000)

def evaluate_model(model, X_val, Y_val):
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        _, predicted_classes = outputs.max(2)#score probabilité  ->indice val max
        correct = (predicted_classes.view(-1) == Y_val.view(-1)).float()# tenseur de valeur boolean vrai ou  faux frais 1 .0faux 0.0
        accuracy = correct.mean()
        print(f'Validation Accuracy: {accuracy.item() * 100:.2f}%')

evaluate_model(model, X_val, Y_val)

def show_examples(model, X, real_y, num_examples=10):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted_classes = outputs.max(2)
        for i in range(num_examples):
            example_input = decode_sudoku_grid(X[i].numpy())
            predicted_output = predicted_classes[i].cpu().numpy().reshape(4, 4) + 1
            real_output = real_y[i].cpu().numpy().reshape(4, 4) + 1

            print(f"Example {i+1} Sudoku Grid (Input):")
            print(example_input)
            print("\nPredicted Sudoku Grid:")
            print(predicted_output)
            print("\nReal Sudoku Grid (Solution):")
            print(real_output)
            print("\n" + "#" * 40 + "\n")

show_examples(model, X_val, Y_val, 10)
