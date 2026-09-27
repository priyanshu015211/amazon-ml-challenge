"""
src/matching.py

Final entity-resolution matching layer.

Pipeline:

    Source 1
       |
       v
    Blocking
       |
       v
    Candidate pairs
       |
       v
    Feature generation
       |
       v
    Random Forest
       |
       v
    Probability
       |
       v
    Decision / No-match
       |
       v
    matching_results.tsv

Expected prediction convention:

    1 = MATCH
    0 = NO MATCH
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# MATCHER
# ============================================================

class EntityMatcher:
    """
    Convert candidate-pair probabilities into final
    entity-resolution matches.
    """

    def __init__(
        self,
        model,
        threshold=None,
        margin=None,
        allow_multiple=False,
    ):
        """
        Parameters
        ----------
        model:
            Trained EntityMatchingModel.

        threshold:
            Probability required for accepting a match.
            If None, use model.threshold.

        margin:
            Optional difference required between the best
            and second-best candidate.

            Example:

                best       = 0.96
                second     = 0.71
                margin     = 0.20

                0.96 - 0.71 = 0.25
                -> accepted

            Set to None to disable margin filtering.

        allow_multiple:
            If True, more than one candidate can be returned
            for the same Source 1 entity.

            If False, only the strongest accepted candidate
            per source is returned.
        """

        self.model = model

        if threshold is None:
            threshold = model.threshold

        self.threshold = float(threshold)

        self.margin = margin

        self.allow_multiple = allow_multiple

    # ========================================================
    # SCORE CANDIDATES
    # ========================================================

    def score_candidates(
        self,
        candidates,
        feature_matrix,
    ):
        """
        Score candidate pairs using the trained model.

        Parameters
        ----------
        candidates:
            DataFrame containing candidate metadata.

        feature_matrix:
            DataFrame containing exactly the numerical
            features expected by the Random Forest.

        Returns
        -------
        DataFrame
            Candidate metadata + match probability.
        """

        if len(candidates) != len(feature_matrix):

            raise ValueError(
                "candidates and feature_matrix "
                "must have the same number of rows."
            )

        candidates = candidates.reset_index(
            drop=True
        ).copy()

        feature_matrix = feature_matrix.reset_index(
            drop=True
        ).copy()

        probabilities = (
            self.model.predict_probability(
                feature_matrix
            )
        )

        candidates[
            "match_probability"
        ] = probabilities

        return candidates

    # ========================================================
    # RANK CANDIDATES
    # ========================================================

    @staticmethod
    def rank_candidates(
        scored_candidates,
    ):
        """
        Rank candidates separately for every Source 1 entity.

        Ranking is performed independently for Source 2 and
        Source 3.
        """

        df = scored_candidates.copy()

        required = {
            "source1_id",
            "candidate_id",
            "source",
            "match_probability",
        }

        missing = required - set(df.columns)

        if missing:

            raise ValueError(
                f"Missing candidate columns: {sorted(missing)}"
            )

        df = df.sort_values(
            [
                "source1_id",
                "source",
                "match_probability",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )

        # Position within each source-specific candidate list.
        df["candidate_rank"] = (
            df.groupby(
                [
                    "source1_id",
                    "source",
                ]
            )
            .cumcount()
            + 1
        )

        # Probability of second-best candidate.
        df["second_best_probability"] = (
            df.groupby(
                [
                    "source1_id",
                    "source",
                ]
            )["match_probability"]
            .shift(-1)
        )

        df[
            "second_best_probability"
        ] = df[
            "second_best_probability"
        ].fillna(0.0)

        df["probability_margin"] = (
            df["match_probability"]
            - df["second_best_probability"]
        )

        return df

    # ========================================================
    # APPLY THRESHOLD
    # ========================================================

    def apply_threshold(
        self,
        ranked_candidates,
    ):
        """
        Apply probability threshold and optional margin.
        """

        df = ranked_candidates.copy()

        # Basic probability condition
        df["passes_threshold"] = (
            df["match_probability"]
            >= self.threshold
        )

        # ----------------------------------------------------
        # Margin condition
        # ----------------------------------------------------

        if self.margin is None:

            df["passes_margin"] = True

        else:

            df["passes_margin"] = (
                df["probability_margin"]
                >= self.margin
            )

        # ----------------------------------------------------
        # Final acceptance
        # ----------------------------------------------------

        df["accepted"] = (
            df["passes_threshold"]
            & df["passes_margin"]
        )

        return df

    # ========================================================
    # SELECT BEST MATCH
    # ========================================================

    def select_matches(
        self,
        scored_candidates,
    ):
        """
        Select final matches.

        Default behavior:
            - One strongest match per Source 1 entity
              per source.
            - No match if no candidate passes the rules.
        """

        if scored_candidates.empty:

            return pd.DataFrame(
                columns=[
                    "source1_id",
                    "candidate_id",
                    "source",
                    "match_probability",
                ]
            )

        df = self.rank_candidates(
            scored_candidates
        )

        df = self.apply_threshold(
            df
        )

        # ----------------------------------------------------
        # Keep accepted candidates only
        # ----------------------------------------------------

        accepted = df[
            df["accepted"]
        ].copy()

        if accepted.empty:

            return pd.DataFrame(
                columns=[
                    "source1_id",
                    "candidate_id",
                    "source",
                    "match_probability",
                ]
            )

        # ----------------------------------------------------
        # One candidate per source
        # ----------------------------------------------------

        if not self.allow_multiple:

            accepted = (
                accepted
                .sort_values(
                    "match_probability",
                    ascending=False,
                )
                .drop_duplicates(
                    subset=[
                        "source1_id",
                        "source",
                    ],
                    keep="first",
                )
            )

        return accepted[
            [
                "source1_id",
                "candidate_id",
                "source",
                "match_probability",
            ]
        ].reset_index(
            drop=True
        )

    # ========================================================
    # FINAL MATCH TABLE
    # ========================================================

    def build_output(
        self,
        source1,
        selected_matches,
    ):
        """
        Create one output row for every Source 1 entity.

        Source 2 and Source 3 matches are stored separately.

        Example:

            source1_id    source2_id    source3_id
            S1-001        S2-100        S3-300
            S1-002        S2-101
            S1-003

        Empty cells mean no match.
        """

        if "id" not in source1.columns:

            raise ValueError(
                "Source 1 must contain an 'id' column."
            )

        # ----------------------------------------------------
        # Start with every Source 1 entity.
        # ----------------------------------------------------

        output = pd.DataFrame({
            "source1_id": source1["id"]
        }).drop_duplicates()

        # ----------------------------------------------------
        # Source 2
        # ----------------------------------------------------

        source2_matches = (
            selected_matches[
                selected_matches["source"]
                == "source2"
            ]
            .sort_values(
                "match_probability",
                ascending=False,
            )
            .drop_duplicates(
                "source1_id"
            )
            [
                [
                    "source1_id",
                    "candidate_id",
                ]
            ]
            .rename(
                columns={
                    "candidate_id":
                        "source2_id"
                }
            )
        )

        # ----------------------------------------------------
        # Source 3
        # ----------------------------------------------------

        source3_matches = (
            selected_matches[
                selected_matches["source"]
                == "source3"
            ]
            .sort_values(
                "match_probability",
                ascending=False,
            )
            .drop_duplicates(
                "source1_id"
            )
            [
                [
                    "source1_id",
                    "candidate_id",
                ]
            ]
            .rename(
                columns={
                    "candidate_id":
                        "source3_id"
                }
            )
        )

        # ----------------------------------------------------
        # Merge results
        # ----------------------------------------------------

        output = output.merge(
            source2_matches,
            on="source1_id",
            how="left",
        )

        output = output.merge(
            source3_matches,
            on="source1_id",
            how="left",
        )

        # Empty strings instead of NaN.
        output["source2_id"] = (
            output["source2_id"]
            .fillna("")
        )

        output["source3_id"] = (
            output["source3_id"]
            .fillna("")
        )

        return output

    # ========================================================
    # RUN COMPLETE MATCHING
    # ========================================================

    def run(
        self,
        source1,
        candidates,
        feature_matrix,
    ):
        """
        Complete inference pipeline.

        Returns
        -------
        output:
            Final Source 1 -> Source 2 / Source 3 table.

        selected:
            Accepted matches.

        scored:
            All candidates with probabilities.
        """

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        scored = self.score_candidates(
            candidates,
            feature_matrix,
        )

        # ----------------------------------------------------
        # Select
        # ----------------------------------------------------

        selected = self.select_matches(
            scored
        )

        # ----------------------------------------------------
        # Final output
        # ----------------------------------------------------

        output = self.build_output(
            source1,
            selected,
        )

        return output, selected, scored

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    @staticmethod
    def save_results(
        output,
        path="output/matching_results.tsv",
    ):
        """
        Save final competition submission.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.to_csv(
            path,
            sep="\t",
            index=False,
        )

        print(
            f"Matching results saved to: {path}"
        )

    # ========================================================
    # SAVE AUDIT RESULTS
    # ========================================================

    @staticmethod
    def save_audit(
        scored,
        selected,
        path="output/matching_audit.tsv",
    ):
        """
        Save candidate probabilities and decisions.

        This is useful for debugging and research analysis.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        audit = scored.copy()

        # ----------------------------------------------------
        # Mark selected candidates
        # ----------------------------------------------------

        if selected.empty:

            audit["selected"] = False

        else:

            selected_keys = set(
                zip(
                    selected["source1_id"],
                    selected["candidate_id"],
                    selected["source"],
                )
            )

            audit["selected"] = [
                (
                    row["source1_id"],
                    row["candidate_id"],
                    row["source"],
                )
                in selected_keys
                for _, row in audit.iterrows()
            ]

        audit.to_csv(
            path,
            sep="\t",
            index=False,
        )

        print(
            f"Audit results saved to: {path}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    @staticmethod
    def summary(
        source1,
        scored,
        selected,
    ):
        """
        Print useful matching statistics.
        """

        total_entities = len(
            source1
        )

        total_candidates = len(
            scored
        )

        total_selected = len(
            selected
        )

        matched_s1 = (
            selected["source1_id"]
            .nunique()
            if not selected.empty
            else 0
        )

        unmatched_s1 = (
            total_entities
            - matched_s1
        )

        print()
        print("=" * 60)
        print("MATCHING SUMMARY")
        print("=" * 60)

        print(
            f"Source 1 entities:     "
            f"{total_entities:,}"
        )

        print(
            f"Candidate pairs:       "
            f"{total_candidates:,}"
        )

        print(
            f"Accepted matches:      "
            f"{total_selected:,}"
        )

        print(
            f"Matched Source 1:      "
            f"{matched_s1:,}"
        )

        print(
            f"Unmatched Source 1:    "
            f"{unmatched_s1:,}"
        )

        if total_entities > 0:

            coverage = (
                matched_s1
                / total_entities
            )

            print(
                f"Match coverage:        "
                f"{coverage:.2%}"
            )

        # ----------------------------------------------------
        # Source-specific statistics
        # ----------------------------------------------------

        if not selected.empty:

            source_counts = (
                selected["source"]
                .value_counts()
            )

            print("\nMatches by source:")

            for source, count in (
                source_counts.items()
            ):

                print(
                    f"  {source}: {count:,}"
                )

        # ----------------------------------------------------
        # Probability statistics
        # ----------------------------------------------------

        if not selected.empty:

            print(
                "\nAccepted probability statistics:"
            )

            print(
                selected[
                    "match_probability"
                ].describe().to_string()
            )
