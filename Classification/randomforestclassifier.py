import itertools
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier


def cross_validation_class_pair_rf(
    df: pd.DataFrame,
    label_column: str,
    n_splits: int = 5
):
    # ------------------------------------------------------------------ #
    # 1.  Set up the single classifier (Random Forest)
    # ------------------------------------------------------------------ #
    # We are now using only RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)


    # ------------------------------------------------------------------ #
    # 2.  Encode labels
    # ------------------------------------------------------------------ #
    df = df.copy()
    label_encoder = LabelEncoder()
    df[label_column] = label_encoder.fit_transform(df[label_column])
    class_mapping = dict(
        zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_)
    )
    classes = df[label_column].unique()

    # ------------------------------------------------------------------ #
    # 3.  Storage
    # ------------------------------------------------------------------ #
    results = []
    misclassified_details = []

    # ------------------------------------------------------------------ #
    # 4.  One‑vs‑one evaluation
    # ------------------------------------------------------------------ #
    for class1, class2 in itertools.combinations(classes, 2):
        # --- keep a copy to save real row IDs --------------------------------
        df_pair = df[df[label_column].isin([class1, class2])].copy()
        df_pair["orig_index"] = df_pair.index  # preserve the real index

        X = df_pair.drop(columns=[label_column, "orig_index"]) # Drop orig_index from features
        y = df_pair[label_column]
        orig_idx = df_pair["orig_index"].reset_index(drop=True)

        # reset feature/label index so StratifiedKFold produces 0…n‑1 integers
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_scores, precs, recalls, f1s, aucs = [], [], [], [], []

        for train_idx, test_idx in skf.split(X, y):
            # ------------------------------------------------------------------
            # split, scale
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # ------------------------------------------------------------------
            # RandomForest predictions and probabilities
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            # Get probability for the positive class (class1)
            # Need to find the index of class1 in the classes array of the fitted classifier
            class1_index = list(clf.classes_).index(class1)
            y_prob = clf.predict_proba(X_test)[:, class1_index]


            # ------------------------------------------------------------------
            # log misclassifications with REAL index + task
            for i, idx in enumerate(test_idx):
                if y_test.iloc[i] != y_pred[i]:
                    misclassified_details.append(
                        {
                            "Index": int(orig_idx.iloc[i]), # Use orig_idx.iloc[i] to get the real index for this test instance
                            "Task": f"{class_mapping[class1]} vs {class_mapping[class2]}",
                            "True Value": int(y_test.iloc[i]),
                            "Predicted Value": int(y_pred[i]),
                        }
                    )

            # ------------------------------------------------------------------
            # metrics for this fold
            fold_scores.append(accuracy_score(y_test, y_pred))
            # Ensure pos_label is one of the classes in the current pair
            # Precision, Recall, F1-Score are calculated for the positive class (class1)
            precs.append(
                precision_score(y_test, y_pred, pos_label=class1, average="binary")
            )
            recalls.append(recall_score(y_test, y_pred, pos_label=class1, average="binary"))
            f1s.append(f1_score(y_test, y_pred, pos_label=class1, average="binary"))
            aucs.append(roc_auc_score(y_test, y_prob))

        # ----------------------------------------------------------------------
        # aggregate pair result
        results.append(
            {
                "Class1": class_mapping[class1],
                "Class2": class_mapping[class2],
                "Mean Accuracy": np.mean(fold_scores),
                "Mean Precision": np.mean(precs),
                "Mean Recall": np.mean(recalls),
                "Mean F1-Score": np.mean(f1s),
                "Mean AUC": np.mean(aucs),
            }
        )

    # ------------------------------------------------------------------ #
    # 5.  Multiclass evaluation
    # ------------------------------------------------------------------ #
    X_full = df.drop(columns=[label_column, "orig_index"]) # Drop orig_index from features
    y_full = df[label_column]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_binarized = label_binarize(y_full, classes=classes)

    m_acc, m_prec, m_rec, m_f1, m_auc = [], [], [], [], []

    for train_idx, test_idx in skf.split(X_full, y_full):
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # RandomForest predictions and probabilities
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test) # Probabilities for all classes

        # Misclassification logging for multiclass - using original index from full dataframe
        for i, idx in enumerate(test_idx):
            # Get the original index from the full dataframe using the test_idx
            original_df_index = df.index[idx] # This gets the real index from the original df
            if y_test.iloc[i] != y_pred[i]:
                misclassified_details.append(
                    {
                        "Index": int(original_df_index),
                        "Task": "Multiclass",
                        "True Value": int(y_test.iloc[i]),
                        "Predicted Value": int(y_pred[i]),
                    }
                )


        m_acc.append(accuracy_score(y_test, y_pred))
        m_prec.append(precision_score(y_test, y_pred, average="weighted"))
        m_rec.append(recall_score(y_test, y_pred, average="weighted"))
        m_f1.append(f1_score(y_test, y_pred, average="weighted"))
        # AUC for multiclass needs the binarized true labels and probabilities
        m_auc.append(
            roc_auc_score(y_binarized[test_idx], y_prob, average="macro", multi_class="ovr")
        )

    results.append(
        {
            "Class1": "Multiclass",
            "Class2": "Multiclass",
            "Mean Accuracy": np.mean(m_acc),
            "Mean Precision": np.mean(m_prec),
            "Mean Recall": np.mean(m_rec),
            "Mean F1-Score": np.mean(m_f1),
            "Mean AUC": np.mean(m_auc),
        }
    )

    # ------------------------------------------------------------------ #
    # 6.  Assemble final DataFrames
    # ------------------------------------------------------------------ #
    results_df = pd.DataFrame(results)

    misclassified_df = pd.DataFrame(misclassified_details)

    # convert encoded integers to class names only if there is something to convert
    if not misclassified_df.empty:
        misclassified_df.replace(
        {"True Value": class_mapping, "Predicted Value": class_mapping},
        inplace=True,
        )
    # reorder columns
        misclassified_df = misclassified_df[
        ["Index", "Task", "True Value", "Predicted Value"]
        ]
    else:
      # create an empty frame with the expected columns
        misclassified_df = pd.DataFrame(
        columns=["Index", "Task", "True Value", "Predicted Value"]
        )


    return results_df, misclassified_df
