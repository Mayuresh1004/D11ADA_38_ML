import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()


# Step 1: Load the dataset
df = pd.read_csv('Pokemon.csv')
df

# Step 2: Drop unnecessary or duplicate columns
df = df.drop(columns=['#'])  # Drop index column

# Step 3: Drop rows with missing values (like missing Type 2)
df.dropna(inplace=True)

# Step 4: Encode target column (Legendary: True -> 1, False -> 0)
df['Legendary'] = df['Legendary'].astype(int)


# Step 5: One-hot encode 'Type 1' and 'Type 2'
df = pd.get_dummies(df, columns=['Type 1', 'Type 2'], drop_first=True)

# Step 6: Select features (independent variables)
features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation'] + \
            [col for col in df.columns if 'Type 1_' in col or 'Type 2_' in col]

X = df[features]
y = df['Legendary']


# Step 7: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# Step 9: Train Logistic Regression model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)



# Step 10: Predictions and Evaluation
y_pred = model.predict(X_test)
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])



print("=== Classification Report ===")
print(f"{'Class':<15}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}")
print(f"{'Not Legendary':<15}{precision[0]:<10.2f}{recall[0]:<10.2f}{f1[0]:<10.2f}")
print(f"{'Legendary':<15}{precision[1]:<10.2f}{recall[1]:<10.2f}{f1[1]:<10.2f}")


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Legendary", "Legendary"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()