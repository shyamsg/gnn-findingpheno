import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import tqdm as notebook_tqdm

features_file = 'new_data/Tomato_MGYS00006231_ssu_input.txt'
labels_file = 'new_data/Tomato_MGYS00006231_ssu_label1.txt'
adjacency_file = 'new_data/gene_closest_distances_Tomato_6231.csv'

features_df = pd.read_csv(features_file, delimiter='\t')
features = torch.tensor(features_df.iloc[:, 1:].values, dtype=torch.float)

labels_df = pd.read_csv(labels_file, delimiter='\t')
label_encoder = LabelEncoder()
labels_numerical = label_encoder.fit_transform(labels_df['Rhizosphere'])
labels = torch.tensor(labels_numerical, dtype=torch.long)

adjacency_df = pd.read_csv(adjacency_file)
gene_names = features_df.columns[1:]

gene1_indices = adjacency_df['Gene1'].apply(lambda x: gene_names.get_loc(x)).values # 
gene2_indices = adjacency_df['Gene2'].apply(lambda x: gene_names.get_loc(x)).values
edges = torch.tensor(np.array([gene1_indices, gene2_indices]), dtype=torch.long)
edge_weights = torch.tensor(adjacency_df['Distance'].values, dtype=torch.float)

X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=0.2, random_state=10)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=10)

def create_edge_index(edge_index, edge_weights, node_count):
    mask = (edge_index[0] < node_count) & (edge_index[1] < node_count)
    return edge_index[:, mask], edge_weights[mask]

edge_index_train, edge_weights_train = create_edge_index(edges, edge_weights, X_train.size(0))
edge_index_val, edge_weights_val = create_edge_index(edges, edge_weights, X_val.size(0))
edge_index_test, edge_weights_test = create_edge_index(edges, edge_weights, X_test.size(0))

train_data = Data(x=X_train, edge_index=edge_index_train, edge_attr=edge_weights_train, y=y_train)
val_data = Data(x=X_val, edge_index=edge_index_val, edge_attr=edge_weights_val, y=y_val)
test_data = Data(x=X_test, edge_index=edge_index_test, edge_attr=edge_weights_test, y=y_test)

train_loader = DataLoader([train_data], batch_size=32, shuffle=True)
val_loader = DataLoader([val_data], batch_size=32, shuffle=False)
test_loader = DataLoader([test_data], batch_size=32, shuffle=False)

class EnhancedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(EnhancedGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def objective(trial):
    hidden_dim1 = trial.suggest_int('hidden_dim1', 32, 128)
    hidden_dim2 = trial.suggest_int('hidden_dim2', 64, 256)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])

    model = EnhancedGCN(input_dim=features.size(1), hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=len(label_encoder.classes_))
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Initialize lists to store losses and accuracies
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(2000):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # Calculate training accuracy
            preds = out.argmax(dim=1)
            correct_train += (preds == data.y).sum().item()
            total_train += data.y.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for data in val_loader:
                out = model(data.x, data.edge_index, data.edge_attr)
                val_loss = criterion(out, data.y)
                total_val_loss += val_loss.item()
                preds = out.argmax(dim=1)
                correct_val += (preds == data.y).sum().item()
                total_val += data.y.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        trial.report(val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            raise optuna.TrialPruned()

    # Plotting training and validation losses and accuracies
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))

    # Plot Training and Validation Loss
    axs[0].plot(train_losses, label='Training Loss')
    axs[0].plot(val_losses, label='Validation Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot Training and Validation Accuracy
    axs[1].plot(train_accuracies, label='Training Accuracy', color='red')
    axs[1].plot(val_accuracies, label='Validation Accuracy', color='teal')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return best_val_loss

study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_params = trial.params
final_model = EnhancedGCN(input_dim=features.size(1), hidden_dim1=best_params['hidden_dim1'], hidden_dim2=best_params['hidden_dim2'], output_dim=len(label_encoder.classes_))
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
final_model.train()

best_model_state = None
best_val_accuracy = 0

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(2000):
    total_train_loss = 0
    correct_val = 0
    total_val = 0

    for data in train_loader:
        final_optimizer.zero_grad()
        out = final_model(data.x, data.edge_index, data.edge_attr)
        loss = nn.CrossEntropyLoss()(out, data.y)
        loss.backward()
        final_optimizer.step()
        total_train_loss += loss.item()

    train_losses.append(total_train_loss / len(train_loader))

    final_model.eval()
    with torch.no_grad():
        for data in val_loader:
            out = final_model(data.x, data.edge_index, data.edge_attr)
            preds = out.argmax(dim=1)
            correct_val += (preds == data.y).sum().item()
            total_val += data.y.size(0)

    val_accuracy = correct_val / total_val
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}, Loss: {train_losses[-1]}, Val Accuracy: {val_accuracy:.4f}')

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = final_model.state_dict()

final_model.load_state_dict(best_model_state)

final_model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for data in test_loader:
        out = final_model(data.x, data.edge_index, data.edge_attr)
        preds = out.argmax(dim=1)
        test_preds.append(preds)
        test_labels.append(data.y)

test_preds = torch.cat(test_preds)
test_labels = torch.cat(test_labels)

test_accuracy = accuracy_score(test_labels.numpy(), test_preds.numpy())
print(f'Test Accuracy: {test_accuracy:.4f}')

confusion_mtx = confusion_matrix(test_labels.numpy(), test_preds.numpy())

plt.figure(figsize=(10, 8))
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(label_encoder.classes_))
plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
plt.yticks(tick_marks, label_encoder.classes_)
plt.ylabel('True label')
plt.xlabel('Predicted label')

thresh = confusion_mtx.max() / 2.0
for i, j in np.ndindex(confusion_mtx.shape):
    plt.text(j, i, f'{confusion_mtx[i, j]}', ha='center', va='center', color='white' if confusion_mtx[i, j] > thresh else 'black')

plt.tight_layout()
plt.show()

# Plotting training and validation loss and accuracy
plt.figure(figsize=(14, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
