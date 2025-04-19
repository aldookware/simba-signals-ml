from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    f1_score,
    precision_score,
    recall_score,
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def safe_f1_macro_score(y_true, y_pred):
    # Safely compute macro F1, adjusting for missing classes in folds
    labels = ['Buy', 'Neutral', 'Sell']
    present = [
        lbl for lbl in labels if lbl in np.unique(np.concatenate((y_true, y_pred)))
    ]
    return f1_score(y_true, y_pred, labels=present, average='macro', zero_division=0)


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
    plt.savefig(output_file)  # Save first, then show
    plt.close()


def train_model(df, default_thresh=0.7, class_threshs=None, save_model=True):
    """
    Trains a RandomForestClassifier using a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with features and Signal column
        conf_thresh (float): Confidence threshold for predictions
        save_model (bool): Whether to save the model to disk

    Returns:
        RandomForestClassifier: The trained model
    """
    try:
        # Split features and labels
        features = df.drop(columns=["Signal"])
        labels = df["Signal"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # 🔧 Simpler tuning grid
        param_grid = {
            'n_estimators': [290, 300, 305, 310],
            'max_depth': [25, 30, 32, 35],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced'],
        }

        base_model = RandomForestClassifier(random_state=42)
        scorer = make_scorer(safe_f1_macro_score, greater_is_better=True)

        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, n_jobs=None, scoring=scorer
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print("Best Params:", grid_search.best_params_)

        # Raw predictions and probabilities
        y_pred_raw = best_model.predict(X_test)
        proba = best_model.predict_proba(X_test)
        # max_proba = np.max(proba, axis=1)

        # Make sure 'Neutral' is in the classes
        if 'Neutral' not in best_model.classes_:
            print(
                "Warning: 'Neutral' class not found, adding it for threshold handling."
            )
            best_model.classes_ = np.append(best_model.classes_, 'Neutral')
        # Apply confidence threshold: set to Neutral if below threshold
        y_pred = []
        for pred, prob in zip(y_pred_raw, proba):
            # pick probability corresponding to predicted class
            idx = list(best_model.classes_).index(pred)
            prob = prob[idx]
            # Get threshold for this prediction class (or use default)
            thresh = (
                class_threshs.get(pred, default_thresh)
                if class_threshs
                else default_thresh
            )

            y_pred.append(pred if prob >= thresh else 'Neutral')
        print(
            f"Thresholds applied – default: {default_thresh}, "
            f"per-class: {class_threshs}"
        )
        print(classification_report(y_test, y_pred))
        # Classification report with filtered predictions
        print(classification_report(y_test, y_pred))

        # Calculate and store F1 score as an attribute of the model
        f1 = f1_score(y_test, y_pred, average='macro', labels=best_model.classes_)
        best_model.test_f1_score = f1

        # Plot confusion matrix
        try:
            plot_confusion_matrix(y_test, y_pred, classes=best_model.classes_)
        except Exception as e:
            print(f"Warning: Could not plot confusion matrix: {e}")

        # Save the model if requested
        if save_model:
            joblib.dump(best_model, "simba_model.pkl")

        return best_model, X_test, y_test
    except Exception as e:
        print(f"Error in model training: {e}")
        # Return a basic model in case of error to avoid complete failure
        model = RandomForestClassifier(random_state=42)
        model.test_f1_score = 0
        return model, (X_test, y_test)


def threshold_sweep(model, X_test, y_test):
    """
    Sweep thresholds 0.5 to 0.9 for Buy and Sell, plot precision-recall curves.
    """
    thresholds = np.arange(0.5, 0.95, 0.05)
    buy_prec = []
    buy_rec = []
    sell_prec = []
    sell_rec = []

    proba = model.predict_proba(X_test)

    # For each threshold
    for thresh in thresholds:
        y_pred = []
        for i, pred_probs in enumerate(proba):
            # Get the class with highest probability
            pred_class = model.classes_[np.argmax(pred_probs)]
            # Get the probability for that class
            max_prob = np.max(pred_probs)

            # Apply threshold
            if max_prob >= thresh:
                y_pred.append(pred_class)
            else:
                y_pred.append('Neutral')

        # Calculate metrics for each class using appropriate average
        # Change from binary to macro or weighted
        buy_precision = precision_score(
            y_test, y_pred, average='macro', labels=['Buy'], zero_division=0
        )
        buy_recall = recall_score(
            y_test, y_pred, average='macro', labels=['Buy'], zero_division=0
        )

        sell_precision = precision_score(
            y_test, y_pred, average='macro', labels=['Sell'], zero_division=0
        )
        sell_recall = recall_score(
            y_test, y_pred, average='macro', labels=['Sell'], zero_division=0
        )

        buy_prec.append(buy_precision)
        buy_rec.append(buy_recall)
        sell_prec.append(sell_precision)
        sell_rec.append(sell_recall)

    # Plot precision-recall curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(thresholds, buy_prec, 'b-', label='Precision')
    plt.plot(thresholds, buy_rec, 'g-', label='Recall')
    plt.title('Buy Signal Metrics vs Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(thresholds, sell_prec, 'b-', label='Precision')
    plt.plot(thresholds, sell_rec, 'g-', label='Recall')
    plt.title('Sell Signal Metrics vs Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('threshold_sweep.png')
    plt.close()
