from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_test, y_pred, classes, output_file="confusion_matrix.png"):
    """
    Plots and saves the confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def train_model(df):
    """
    Trains a RandomForestClassifier using a DataFrame.
    """
    # Split features and labels
    features = df.drop(columns=["Signal"])
    labels = df["Signal"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=model.classes_)

    # Save the model
    joblib.dump(model, "simba_model.pkl")
    return model
