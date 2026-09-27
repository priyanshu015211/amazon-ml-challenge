
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.metrics import make_scorer


# ============================================================
# F0.5 SCORER
# ============================================================

F05_SCORER = make_scorer(
    fbeta_score,
    beta=0.5,
    zero_division=0,
)


# ============================================================
# MODEL
# ============================================================

class EntityMatchingModel:
    """
    Random Forest model for entity matching.

    Label convention:

        1 = MATCH
        0 = NON-MATCH
    """

    def __init__(self, random_state=42):

        self.random_state = random_state

        self.model = None

        self.feature_columns = None

        self.threshold = 0.5

        self.best_params = None

        self.validation_results = None

    # ========================================================
    # BUILD RANDOM FOREST
    # ========================================================

    def build_model(self, **params):
        """
        Build a Random Forest with sensible defaults.

        Parameters can override the defaults.
        """

        defaults = {
            "n_estimators": 700,

            "max_depth": None,

            "min_samples_split": 4,

            "min_samples_leaf": 2,

            "max_features": "sqrt",

            "class_weight": "balanced_subsample",

            "bootstrap": True,

            "max_samples": 0.9,

            "n_jobs": -1,

            "random_state": self.random_state,
        }

        defaults.update(params)

        return RandomForestClassifier(
            **defaults
        )

    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    def prepare_features(
        self,
        X,
        training=False,
    ):
        """
        Validate and prepare the feature matrix.
        """

        X = X.copy()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        if training:

            self.feature_columns = list(
                X.columns
            )

            if not self.feature_columns:

                raise ValueError(
                    "No feature columns supplied."
                )

            if X.columns.duplicated().any():

                duplicated = (
                    X.columns[
                        X.columns.duplicated()
                    ].tolist()
                )

                raise ValueError(
                    "Duplicate feature columns: "
                    f"{duplicated}"
                )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        else:

            if self.feature_columns is None:

                raise ValueError(
                    "Model has not been trained or loaded."
                )

            missing = [
                column
                for column in self.feature_columns
                if column not in X.columns
            ]

            if missing:

                raise ValueError(
                    "Missing model features: "
                    f"{missing}"
                )

            # Keep exactly the training order
            X = X[
                self.feature_columns
            ]

        # ----------------------------------------------------
        # Convert numeric
        # ----------------------------------------------------

        for column in X.columns:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            )

        # Replace infinity
        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Random Forest can work with missing values in
        # some sklearn versions, but explicit imputation
        # keeps behavior consistent.
        X = X.fillna(-1.0)

        return X.astype(
            np.float32
        )

    # ========================================================
    # HYPERPARAMETER OPTIMIZATION
    # ========================================================

    def optimize_hyperparameters(
        self,
        X,
        y,
        groups,
        n_iter=25,
        cv=3,
    ):
        """
        Search for a strong Random Forest configuration.

        Group-aware CV prevents candidate pairs belonging
        to the same Source 1 entity from leaking between
        folds.
        """

        if len(
            np.unique(y)
        ) < 2:

            raise ValueError(
                "Hyperparameter optimization requires "
                "both positive and negative examples."
            )

        # ----------------------------------------------------
        # Base estimator
        # ----------------------------------------------------

        base_model = self.build_model(
            n_jobs=1
        )

        # ----------------------------------------------------
        # Search space
        # ----------------------------------------------------

        parameter_grid = {

            "n_estimators": [
                400,
                600,
                800,
                1000,
            ],

            "max_depth": [
                12,
                20,
                30,
                40,
                None,
            ],

            "min_samples_split": [
                2,
                4,
                8,
                12,
            ],

            "min_samples_leaf": [
                1,
                2,
                4,
                8,
            ],

            "max_features": [
                "sqrt",
                "log2",
                0.5,
                0.8,
            ],

            "class_weight": [
                "balanced",
                "balanced_subsample",
                None,
            ],

            "max_samples": [
                0.7,
                0.85,
                0.9,
                1.0,
            ],
        }

        # ----------------------------------------------------
        # Group-aware cross validation
        # ----------------------------------------------------

        group_cv = StratifiedGroupKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state,
        )

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        search = RandomizedSearchCV(

            estimator=base_model,

            param_distributions=parameter_grid,

            n_iter=n_iter,

            scoring=F05_SCORER,

            cv=group_cv,

            n_jobs=-1,

            verbose=1,

            random_state=self.random_state,

            refit=True,
        )

        search.fit(
            X,
            y,
            groups=groups,
        )

        self.best_params = (
            search.best_params_
        )

        print()
        print("=" * 60)
        print("RANDOM FOREST OPTIMIZATION")
        print("=" * 60)

        print("\nBest parameters:")

        for key, value in (
            self.best_params.items()
        ):

            print(
                f"  {key}: {value}"
            )

        print(
            "\nBest CV F0.5:",
            round(
                search.best_score_,
                5,
            ),
        )

        return self.best_params

    # ========================================================
    # HARD NEGATIVE SAMPLING
    # ========================================================

    @staticmethod
    def sample_hard_negatives(
        df,
        label_column="label",
        similarity_column="name_core_ratio",
        negative_ratio=5,
        hard_fraction=0.7,
        random_state=42,
    ):
        """
        Select informative negative examples.

        IMPORTANT:
        This function should only be applied to a dataset
        where label=0 means a verified non-match.

        It should NOT treat every unlabeled candidate as
        a negative.
        """

        if label_column not in df.columns:

            raise ValueError(
                f"Missing label column: "
                f"{label_column}"
            )

        if (
            similarity_column
            not in df.columns
        ):

            raise ValueError(
                f"Missing similarity column: "
                f"{similarity_column}"
            )

        positives = df[
            df[label_column] == 1
        ].copy()

        negatives = df[
            df[label_column] == 0
        ].copy()

        if positives.empty:

            raise ValueError(
                "No positive examples found."
            )

        if negatives.empty:

            raise ValueError(
                "No negative examples found."
            )

        target_negative_count = min(
            len(negatives),
            len(positives)
            * negative_ratio,
        )

        hard_count = int(
            target_negative_count
            * hard_fraction
        )

        # ----------------------------------------------------
        # Hard negatives
        # ----------------------------------------------------

        hard_negatives = (
            negatives
            .sort_values(
                similarity_column,
                ascending=False,
            )
            .head(hard_count)
        )

        # ----------------------------------------------------
        # Random negatives
        # ----------------------------------------------------

        remaining = negatives.drop(
            index=hard_negatives.index
        )

        random_count = (
            target_negative_count
            - len(hard_negatives)
        )

        random_count = min(
            random_count,
            len(remaining),
        )

        if random_count > 0:

            random_negatives = (
                remaining.sample(
                    n=random_count,
                    random_state=random_state,
                )
            )

        else:

            random_negatives = (
                remaining.iloc[0:0]
            )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        result = pd.concat(
            [
                positives,
                hard_negatives,
                random_negatives,
            ],
            ignore_index=True,
        )

        result = result.sample(
            frac=1.0,
            random_state=random_state,
        ).reset_index(
            drop=True
        )

        return result

    # ========================================================
    # THRESHOLD OPTIMIZATION
    # ========================================================

    def optimize_threshold(
        self,
        y_true,
        probabilities,
        minimum_precision=None,
        min_predicted_matches=1,
    ):
        """
        Find the threshold maximizing F0.5.

        Optionally enforce a minimum precision.

        This is important because this challenge penalizes
        false merges more heavily than false negatives.
        """

        y_true = np.asarray(
            y_true,
            dtype=int,
        )

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        results = []

        thresholds = np.linspace(
            0.01,
            0.99,
            99,
        )

        for threshold in thresholds:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            predicted_matches = int(
                predictions.sum()
            )

            if (
                predicted_matches
                < min_predicted_matches
            ):
                continue

            precision = precision_score(
                y_true,
                predictions,
                zero_division=0,
            )

            recall = recall_score(
                y_true,
                predictions,
                zero_division=0,
            )

            f05 = fbeta_score(
                y_true,
                predictions,
                beta=0.5,
                zero_division=0,
            )

            true_positive = int(
                (
                    (predictions == 1)
                    & (y_true == 1)
                ).sum()
            )

            false_positive = int(
                (
                    (predictions == 1)
                    & (y_true == 0)
                ).sum()
            )

            false_negative = int(
                (
                    (predictions == 0)
                    & (y_true == 1)
                ).sum()
            )

            results.append({

                "threshold": float(
                    threshold
                ),

                "precision": float(
                    precision
                ),

                "recall": float(
                    recall
                ),

                "f0.5": float(
                    f05
                ),

                "predicted_matches":
                    predicted_matches,

                "true_positives":
                    true_positive,

                "false_positives":
                    false_positive,

                "false_negatives":
                    false_negative,
            })

        if not results:

            raise ValueError(
                "No valid threshold found."
            )

        results_df = pd.DataFrame(
            results
        )

        # ----------------------------------------------------
        # Optional precision constraint
        # ----------------------------------------------------

        eligible = results_df.copy()

        if minimum_precision is not None:

            eligible = eligible[
                eligible["precision"]
                >= minimum_precision
            ]

            if eligible.empty:

                raise ValueError(
                    "No threshold achieved the "
                    "requested minimum precision."
                )

        # ----------------------------------------------------
        # Select threshold
        # ----------------------------------------------------

        best = (
            eligible
            .sort_values(
                [
                    "f0.5",
                    "precision",
                    "recall",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

        self.threshold = float(
            best["threshold"]
        )

        self.validation_results = (
            results_df
        )

        return best

    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        X,
        y,
        groups,
        optimize=True,
        n_iter=25,
        cv=3,
        minimum_precision=None,
    ):
        """
        Train the complete Random Forest pipeline.

        Steps:

            1. Prepare features
            2. Create group-aware validation split
            3. Optimize hyperparameters
            4. Train on training partition
            5. Find F0.5-optimal threshold
            6. Report validation performance
            7. Retrain final model on all labeled data
        """

        # ----------------------------------------------------
        # Prepare data
        # ----------------------------------------------------

        X = self.prepare_features(
            X,
            training=True,
        )

        y = np.asarray(
            y,
            dtype=int,
        )

        groups = np.asarray(
            groups
        )

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if not (
            len(X)
            == len(y)
            == len(groups)
        ):

            raise ValueError(
                "X, y and groups must have "
                "the same number of rows."
            )

        if set(
            np.unique(y)
        ) != {0, 1}:

            raise ValueError(
                "Labels must contain both "
                "0 and 1."
            )

        if len(
            np.unique(groups)
        ) < 2:

            raise ValueError(
                "At least two groups are required "
                "for group-aware validation."
            )

        # ----------------------------------------------------
        # Group-aware validation split
        # ----------------------------------------------------

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=self.random_state,
        )

        train_idx, valid_idx = next(
            splitter.split(
                X,
                y,
                groups,
            )
        )

        X_train = X.iloc[
            train_idx
        ]

        X_valid = X.iloc[
            valid_idx
        ]

        y_train = y[
            train_idx
        ]

        y_valid = y[
            valid_idx
        ]

        groups_train = groups[
            train_idx
        ]

        # ----------------------------------------------------
        # Check classes
        # ----------------------------------------------------

        if len(
            np.unique(y_train)
        ) < 2:

            raise ValueError(
                "Training partition contains "
                "only one class."
            )

        if len(
            np.unique(y_valid)
        ) < 2:

            raise ValueError(
                "Validation partition contains "
                "only one class."
            )

        # ----------------------------------------------------
        # Hyperparameter optimization
        # ----------------------------------------------------

        if optimize:

            params = (
                self.optimize_hyperparameters(
                    X_train,
                    y_train,
                    groups_train,
                    n_iter=n_iter,
                    cv=cv,
                )
            )

        else:

            params = {}

        # ----------------------------------------------------
        # Train validation model
        # ----------------------------------------------------

        validation_model = (
            self.build_model(
                **params
            )
        )

        validation_model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Validation probabilities
        # ----------------------------------------------------

        validation_probabilities = (
            validation_model
            .predict_proba(
                X_valid
            )[:, 1]
        )

        # ----------------------------------------------------
        # Optimize threshold
        # ----------------------------------------------------

        best = (
            self.optimize_threshold(
                y_true=y_valid,
                probabilities=(
                    validation_probabilities
                ),
                minimum_precision=(
                    minimum_precision
                ),
            )
        )

        validation_predictions = (
            validation_probabilities
            >= self.threshold
        ).astype(int)

        # ----------------------------------------------------
        # Validation report
        # ----------------------------------------------------

        precision = precision_score(
            y_valid,
            validation_predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_valid,
            validation_predictions,
            zero_division=0,
        )

        f05 = fbeta_score(
            y_valid,
            validation_predictions,
            beta=0.5,
            zero_division=0,
        )

        cm = confusion_matrix(
            y_valid,
            validation_predictions,
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("ENTITY MATCHING VALIDATION")
        print("=" * 60)

        print(
            f"Training rows:   {len(X_train):,}"
        )

        print(
            f"Validation rows: {len(X_valid):,}"
        )

        print(
            f"Training groups: "
            f"{len(np.unique(groups_train)):,}"
        )

        print()
        print(
            f"Threshold: {self.threshold:.3f}"
        )

        print(
            f"Precision: {precision:.5f}"
        )

        print(
            f"Recall:    {recall:.5f}"
        )

        print(
            f"F0.5:      {f05:.5f}"
        )

        print()
        print(
            "Confusion matrix:"
        )

        print(cm)

        print()
        print(
            "Classification report:"
        )

        print(
            classification_report(
                y_valid,
                validation_predictions,
                zero_division=0,
            )
        )

        print()
        print(
            "Selected threshold:"
        )

        print(
            best.to_string()
        )

        # ----------------------------------------------------
        # Store validation information
        # ----------------------------------------------------

        self.validation_summary = {

            "threshold":
                self.threshold,

            "precision":
                precision,

            "recall":
                recall,

            "f0.5":
                f05,

            "confusion_matrix":
                cm.tolist(),

            "train_rows":
                len(X_train),

            "validation_rows":
                len(X_valid),

        }

        # ----------------------------------------------------
        # FINAL MODEL
        #
        # Retrain using ALL labeled data.
        # ----------------------------------------------------

        print()
        print(
            "=" * 60
        )

        print(
            "TRAINING FINAL RANDOM FOREST"
        )

        print(
            "=" * 60
        )

        self.model = self.build_model(
            **params
        )

        self.model.fit(
            X,
            y,
        )

        print(
            f"\nFinal model trained on "
            f"{len(X):,} labeled pairs."
        )

        return self

    # ========================================================
    # PREDICT PROBABILITY
    # ========================================================

    def predict_probability(self, X):
        """
        Return probability that each pair is a match.
        """

        if self.model is None:

            raise ValueError(
                "Model has not been trained "
                "or loaded."
            )

        X = self.prepare_features(
            X,
            training=False,
        )

        probabilities = (
            self.model
            .predict_proba(X)[:, 1]
        )

        return probabilities

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        X,
        threshold=None,
    ):
        """
        Predict MATCH / NON-MATCH.

        Returns:
            1 = MATCH
            0 = NON-MATCH
        """

        probabilities = (
            self.predict_probability(X)
        )

        if threshold is None:

            threshold = self.threshold

        return (
            probabilities >= threshold
        ).astype(int)

    # ========================================================
    # PREDICTION DATAFRAME
    # ========================================================

    def predict_dataframe(
        self,
        X,
    ):
        """
        Return probabilities and predictions
        in a convenient dataframe.
        """

        probabilities = (
            self.predict_probability(X)
        )

        predictions = (
            probabilities
            >= self.threshold
        ).astype(int)

        return pd.DataFrame({

            "match_probability":
                probabilities,

            "prediction":
                predictions,

        })

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    def feature_importance(self):
        """
        Return Random Forest feature importance.
        """

        if self.model is None:

            raise ValueError(
                "Model has not been trained."
            )

        importance = pd.DataFrame({

            "feature":
                self.feature_columns,

            "importance":
                self.model.feature_importances_,

        })

        return (
            importance
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path="models/entity_matcher.joblib",
    ):
        """
        Save the complete model configuration.
        """

        if self.model is None:

            raise ValueError(
                "Cannot save an untrained model."
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        package = {

            "model":
                self.model,

            "threshold":
                self.threshold,

            "feature_columns":
                self.feature_columns,

            "best_params":
                self.best_params,

            "validation_results":
                self.validation_results,

            "validation_summary":
                getattr(
                    self,
                    "validation_summary",
                    None,
                ),

            "random_state":
                self.random_state,
        }

        joblib.dump(
            package,
            path,
        )

        print(
            f"Model saved to: {path}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    @classmethod
    def load(
        cls,
        path,
    ):
        """
        Load a previously trained model.
        """

        package = joblib.load(
            path
        )

        instance = cls(
            random_state=package.get(
                "random_state",
                42,
            )
        )

        instance.model = (
            package["model"]
        )

        instance.threshold = (
            package["threshold"]
        )

        instance.feature_columns = (
            package["feature_columns"]
        )

        instance.best_params = (
            package.get(
                "best_params"
            )
        )

        instance.validation_results = (
            package.get(
                "validation_results"
            )
        )

        instance.validation_summary = (
            package.get(
                "validation_summary"
            )
        )

        print(
            f"Model loaded from: {path}"
        )

        print(
            f"Threshold: "
            f"{instance.threshold:.3f}"
        )

        return instance
