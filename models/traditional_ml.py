import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# Load PyTorch tensors and convert them to NumPy arrays for scikit-learn
X_train_video = torch.load('data/train_video.pt').numpy()
y_train = torch.load('data/train_labels.pt').numpy()
X_val_video = torch.load('data/val_video.pt').numpy()
y_val = torch.load('data/val_labels.pt').numpy()

# Flatten tensors if they are multi-dimensional
X_train = X_train_video.reshape(X_train_video.shape[0], -1)
X_val = X_val_video.reshape(X_val_video.shape[0], -1)

# Initialize and train a Classical ML Model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
preds = rf_model.predict(X_val)
print("Random Forest F1-Score:", f1_score(y_val, preds, average='weighted'))
print(classification_report(y_val, preds))
