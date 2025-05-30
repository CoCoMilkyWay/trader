import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)
import torch
import torch.nn as nn
import torch.optim as optim

# print in full
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

weeks = 12

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tsfresh_features_and_label_{weeks}weeks.parquet")
df = pd.read_parquet(filepath)
print(df.head().transpose())

# | σ    | Cumulative Probability | Approx. % within ±σ range          |
# |------|------------------------|------------------------------------|
# | 0.1  | 0.0797                 | ~7.97% within ±0.1σ                |
# | 0.2  | 0.1587                 | ~15.87% within ±0.2σ               |
# | 0.3  | 0.2266                 | ~22.66% within ±0.3σ               |
# | 0.4  | 0.3108                 | ~31.08% within ±0.4σ               |
# | 0.5  | 0.3829                 | ~38.29% within ±0.5σ               |
# | 1.0  | 0.6827                 | ~68.27% within ±1σ (1-sigma rule)  |
# | 1.5  | 0.8664                 | ~86.64% within ±1.5σ               |
# | 2.0  | 0.9545                 | ~95.45% within ±2σ (2-sigma rule)  |
# | 2.5  | 0.9876                 | ~98.76% within ±2.5σ               |
label = df['label']
sigma = label.std()*0.5  # assume normal distribution (actually scaled version)

def classify(y):
    if y < -sigma:
        return 0
    elif y > sigma:
        return 2
    else:
        return 1

# Feature Label Split
X = df.drop(columns=['label']).values
y = df['label'].apply(classify).values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=np.array(y))

# Standardize features
total_features = X_train.shape[1]
scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# Dimensionality reduction with PCA (optional but recommended)
n_components = min(100, total_features)  # adjust or tune
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# Convert to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32).to(device)
# For classification, ensure labels are LongTensor
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Define a shallow, wide NN
class ShallowWideNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super(ShallowWideNN, self).__init__()
        
        dropout_rate = 0.2 # 0.1~0.5(shallow to deep)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        return self.net(x)


input_dim = X_train_pca.shape[1]
hidden_dim = 64  # wide layer; adjust as needed
n_classes = len(np.unique(np.asarray(y)))
model = ShallowWideNN(input_dim, hidden_dim, n_classes).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 200
batch_size = 64 # would impact model performance by batchnorm1d

# Utility to batch data
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

# Training loop
model.train()
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    epoch_loss /= len(train_dataset)
    if epoch % 1 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

# Evaluation on test set
model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred = np.argmax(probs, axis=1)

# Collect statistics
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, average='weighted')
test_recall = recall_score(y_test, y_pred, average='weighted')
test_f1 = f1_score(y_test, y_pred, average='weighted')
# ROC AUC for multiclass (one-vs-rest)
test_auc = roc_auc_score(y_test, probs, multi_class='ovr')

# Print results
print("Test Results:")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1-score : {test_f1:.4f}")
print(f"ROC AUC  : {test_auc:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 13. PCA explained variance (optional insight)
explained_variance = np.sum(pca.explained_variance_ratio_)
print(f"PCA explained variance by {n_components} components: {explained_variance:.4f}")
