import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# STEP 1: LOAD AND SLICE EXACT ROWS
# ==========================================
print("Loading data...")
df_new = pd.read_csv("C:/Users/salah/Desktop/excel/new1.csv")
df_old = pd.read_csv("C:/Users/salah/Desktop/excel/old1.csv")

# Extract the exact 10-minute experimental blocks
# (We add +1 to the end index because Python slicing stops right before the last number)
print("Slicing manual row blocks...")
df_new_valid = df_new.iloc[3374:7280].copy()
df_old_valid = df_old.iloc[890:4470].copy()

# ==========================================
# STEP 2: FILTER AIR CUTTING
# ==========================================
print("Filtering active cutting phases (Feed > 500)...")
active_new = df_new_valid[df_new_valid['actFeedRate'] > 500].copy()
active_old = df_old_valid[df_old_valid['actFeedRate'] > 500].copy()

# Reset index
active_new.reset_index(drop=True, inplace=True)
active_old.reset_index(drop=True, inplace=True)

print(f"Active Cutting Rows -> New Tool: {len(active_new)}, Worn Tool: {len(active_old)}")


# ==========================================
# STEP 3: FEATURE ENGINEERING (5-SECOND WINDOWS)
# ==========================================
def extract_features(df, window_size=50):
    features = []
    num_windows = len(df) // window_size

    for i in range(num_windows):
        window = df.iloc[i * window_size: (i + 1) * window_size]
        fingerprint = {
            'Torque_Mean': window['aaTorque'].mean(),
            'Torque_Std': window['aaTorque'].std(),
            'Load_Mean': window['aaLoad'].mean(),
            'driveLoad' : window['actSpeedRel'].std(),
            #'driveLoad': window['actSpeedRel'].mean(),

            'Load_Std': window['aaLoad'].std(),
            #'Power_RMS': np.sqrt((window['aaPower'] ** 2).mean())
        }
        features.append(fingerprint)
    return pd.DataFrame(features)


print("Extracting features...")
features_new = extract_features(active_new)
features_worn = extract_features(active_old)

print(f"Total AI Windows -> New: {len(features_new)}, Worn: {len(features_worn)}")

# ==========================================
# STEP 4: LABEL, SPLIT, AND TRAIN
# ==========================================
features_new['Label'] = 0  # 0 = Healthy
features_worn['Label'] = 1  # 1 = Worn

# Combine into master dataset
master_data = pd.concat([features_new, features_worn], ignore_index=True)

X = master_data.drop('Label', axis=1)
y = master_data['Label']

# 70% Training / 30% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ==========================================
# STEP 5: EVALUATION
# ==========================================
predictions = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)

print("\n--- MODEL RESULTS ---")
print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test, predictions, target_names=['New Tool', 'Worn Tool']))

# Optional: Plot Confusion Matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted New', 'Predicted Worn'],
            yticklabels=['Actually New', 'Actually Worn'])
plt.title('Confusion Matrix: Tool Wear Classification', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_final.png', dpi=300)
plt.show()
# ==========================================
# STEP 6: EXTRACT FEATURE IMPORTANCE ("WEIGHTS")
# ==========================================
print("\n--- FEATURE IMPORTANCE ---")
# Extract the importance scores from the trained model
importances = clf.feature_importances_
feature_names = X.columns

# Create a DataFrame to view them easily
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort them from most important to least important
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Print the results as percentages
for index, row in feature_df.iterrows():
    print(f"{row['Feature']}: {row['Importance'] * 100:.2f}%")

# Generate and Save a Bar Chart for the Thesis
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
plt.title('Feature Importance (Which parameters the AI used most)', fontsize=14, weight='bold')
plt.xlabel('Relative Importance (0 to 1)')
plt.ylabel('Sensor Feature')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# ==========================================
# STEP 7: EXTRACT PREDICTION PROBABILITIES
# ==========================================
print("\n--- SAMPLE PREDICTION PROBABILITIES ---")
# Get the raw percentage confidence for the first 5 test windows
probabilities = clf.predict_proba(X_test)

for i in range(5):
    prob_new = probabilities[i][0] * 100
    prob_worn = probabilities[i][1] * 100
    actual_label = "Worn" if y_test.iloc[i] == 1 else "New"

    print(f"Window {i + 1} (Actual: {actual_label}) -> AI Confidence: {prob_new:.1f}% New, {prob_worn:.1f}% Worn")
