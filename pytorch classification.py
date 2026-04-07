import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ==========================================
# STEP 1: LOAD AND SLICE EXACT ROWS
# ==========================================
# Reading the Data frames
df_new = pd.read_csv("C:/Users/salah/Desktop/excel/new1.csv")
df_old = pd.read_csv("C:/Users/salah/Desktop/excel/old1.csv")
# slicing row blocks manually

df_new_valid = df_new.iloc[3374:7280].copy()
df_old_valid = df_old.iloc[890:4470].copy()
# ==========================================
# STEP 2: FILTER AIR CUTTING OUT
# ==========================================
# filtering the active rows
active_new = df_new_valid[df_new_valid["actFeedRate"] == 1852].copy() # it was > 500
active_old = df_old_valid[df_old_valid["actFeedRate"] == 1852].copy()
# reset the index

active_old.reset_index(drop=True, inplace=True)
active_new.reset_index(drop=True, inplace=True)
# ==========================================
# STEP 3: FEATURE ENGINEERING (5-SECOND WINDOWS)
# ==========================================
# Feature extraction function

def extract(df, window_size=50):
    features = []
    num_windows = len(df) // window_size
    for i in range(num_windows):
        window = df[i * window_size:(i + 1) * window_size]
        finger_print = {
            'Torque_Mean': window['aaTorque'].mean(),
            'Torque_Std': window['aaTorque'].std(),
        'Load_Mean': window['aaLoad'].mean(),
        'driveLoad': window['actSpeedRel'].std(),
        # 'driveLoad': window['actSpeedRel'].mean(),

        'Load_Std': window['aaLoad'].std(),
        # 'Power_RMS': np.sqrt((window['aaPower'] ** 2).mean())


        }
        features.append(finger_print)
    return pd.DataFrame(features)
# Extracting the fetures
features_new = extract(active_new)
features_old = extract(active_old)

#print(f"Total AI Windows -> New: {len(features_new)}, Worn: {len(features_old)}")
# ===========
#Sub Step define the accuracy function
#============
def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


# ==========================================
# STEP 4: LABEL, CONVERT TO TENSOR, SPLIT, AND TRAIN
# ==========================================
# Label, Split and train
features_new['Label'] = 0 # 0 = Healthy
features_old['Label'] = 1 # 1 = Worn

# Combine the Data
master_data = pd.concat([features_new, features_old], ignore_index=True)

# Set Data into  y and X
y = master_data['Label']
X = master_data.drop('Label', axis=1)
# Convert Data into tensors, Important
X = torch.from_numpy(X.values).type(torch.float)


y = torch.from_numpy(y.values).type(torch.float)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED) # Change the percentage of the training data
# Train
# Making Device agnostic code, fancy for "Get a GPU, or cry about not having it"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model_0 = nn.Sequential(
    nn.Linear(5,32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),

    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Linear(4,1),

).to(device)

untraind_predictions = model_0(X_train.to(device))
print(len(untraind_predictions))
print(len(y_train))
print(torch.sigmoid(untraind_predictions))
# Set the loss function and the manual seed
loss_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.0008)
# The actual training


epochs = 5000
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_func(y_logits, y_train)
    acc = accuracy(y_train, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_func(test_logits, y_test)
        test_acc = accuracy(y_test, test_pred)
    if epoch % 100 ==0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")



