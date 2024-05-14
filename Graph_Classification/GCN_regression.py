import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

features_file = 'scaled_input.csv'
labels_file = 'final_input.csv'

features_df = pd.read_csv(features_file)
features = torch.tensor(features_df.iloc[:, 1:].values, dtype=torch.float)
sample_ids = features_df.iloc[:, 0].values

labels_df = pd.read_csv(labels_file)
labels = torch.tensor(labels_df['weight'].values, dtype=torch.float)

adjacency_file = 'gene_closest_distances.csv'
adjacency_df = pd.read_csv(adjacency_file)

gene_names = features_df.columns[1:]
gene1_indices = adjacency_df['Gene1'].apply(lambda x: gene_names.get_loc(x)).values
gene2_indices = adjacency_df['Gene2'].apply(lambda x: gene_names.get_loc(x)).values
distances = adjacency_df['Distance'].values

edges = torch.tensor([gene1_indices, gene2_indices], dtype=torch.long)
edge_attr = torch.tensor(distances, dtype=torch.float)

X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25)

train_data = Data(x=X_train, edge_index=edges, edge_attr=edge_attr, y=y_train)
val_data = Data(x=X_val, edge_index=edges, edge_attr=edge_attr, y=y_val)
test_data = Data(x=X_test, edge_index=edges, edge_attr=edge_attr, y=y_test)

train_loader = DataLoader([train_data], batch_size=64, shuffle=True)
val_loader = DataLoader([val_data], batch_size=64, shuffle=False)
test_loader = DataLoader([test_data], batch_size=64, shuffle=False)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.fc = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        #x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


model = GCN(input_dim=features.size(1), hidden_dim1=64, hidden_dim2=128, hidden_dim3=64, output_dim=1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loss = []
validation_loss = []
best_val_loss = np.inf
random_loss_list = []

for epoch in range(500):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    #print(f'Epoch {epoch+1}, Loss: {total_loss}')

    model.eval()
    val_loss = 0
    random_loss = 0
    with torch.no_grad():
        for data in val_loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            val_loss += criterion(out.squeeze(), data.y).item()
            shuffled_y = data.y[torch.randperm(data.y.size()[0])] #torch.randperm(data.y)
            random_loss += criterion(shuffled_y, data.y).item()

    print(f'Epoch {epoch+1}, Train Loss: {total_loss}, Validation Loss: {val_loss}')

    train_loss.append(total_loss)
    validation_loss.append(val_loss)
    random_loss_list.append(random_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss

test_loss = 0
with torch.no_grad():
    preds = []
    for data in test_loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.append(out)
        test_loss += criterion(out.squeeze(), data.y).item()

mean_abs_error = mean_absolute_error(out, data.y)

print(f"Final MAE  {best_val_loss}")
print(f'Best val loss {best_val_loss} End Test Loss: {test_loss}')
print(f'Best val loss sqrt {np.sqrt(best_val_loss)} End Test Loss sqrt: {np.sqrt(test_loss)}')

plt.plot(train_loss, label="Train", color='blue')
plt.plot(validation_loss, label="Validation", color='red')
plt.plot(random_loss_list, label="Random", color='k')
plt.legend()
plt.show()

plt.scatter(x =  data.y, y = out)
plt.axline((0, 0), slope=1., color='C0', label='unity')
plt.xlabel("True")
plt.ylabel("Predicted")
plt.show()
