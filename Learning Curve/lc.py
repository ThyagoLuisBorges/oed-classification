import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve


def plot_learning_curve_rf(
    df: pd.DataFrame,
    label_column: str,
    rf_params: dict | None = None,
    n_splits: int = 5,
    train_sizes: np.ndarray | None = None,
    scoring: str = "balanced_accuracy",
    figsize: tuple[int, int] = (8, 5),
):
    # 1 — prepare data --------------------------------------------------------
    df = df.copy()
    le = LabelEncoder()
    y = le.fit_transform(df[label_column])
    X = df.drop(columns=[label_column])

    # 2 — estimator pipeline (scaling + RF) -----------------------------------
    if rf_params is None:
        rf_params = {"n_estimators": 200, "random_state": 42}

    est = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("rf", RandomForestClassifier(**rf_params)),
        ]
    )

    # 3 — learning-curve computation ------------------------------------------
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    # StratifiedKFold handles shuffling and random_state for cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_sizes_abs, train_scores, valid_scores = learning_curve(
        estimator=est,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    valid_std = valid_scores.std(axis=1)

    # 4 — plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train_sizes_abs, train_mean, marker="o", label="Training", lw=2)
    ax.fill_between(
        train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2
    )
    ax.plot(train_sizes_abs, valid_mean, marker="o", label="Cross-validation", lw=2)
    ax.fill_between(
        train_sizes_abs, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2
    )

    ax.set_xlabel("Training examples")
    ax.set_ylabel(scoring.replace("_", " ").title())
    ax.set_title(f"Learning curve – RandomForest ({scoring})")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Return raw numbers in case you want to reuse them
    return {
        "train_sizes": train_sizes_abs,
        "train_scores": train_scores,
        "valid_scores": valid_scores,
    }