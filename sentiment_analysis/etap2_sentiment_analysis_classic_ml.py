import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# Load preprocessed data
df = pd.read_csv('../csv_files/02_preprocessed_data/preprocessed_data.tsv', sep='\t')
df['avg_vector'] = df['avg_vector'].apply(lambda x: ast.literal_eval(x))

# Split data into training and testing sets
X = np.vstack(df['avg_vector'].values)
y = df['cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose three machine learning models
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(label_binarize(y_test, classes=[0, 1, 2]), label_binarize(y_pred, classes=[0, 1, 2]),
                            average='macro')
    fpr, tpr, _ = roc_curve(label_binarize(y_test, classes=[0, 1, 2]).ravel(),
                            label_binarize(y_pred, classes=[0, 1, 2]).ravel())

    # Print results
    print(f"{model_name}\nConfusion Matrix:\n{cm}\nROC AUC Score: {roc_auc}")

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

# Show the ROC curve plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
