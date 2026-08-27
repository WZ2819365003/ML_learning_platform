"""Fusion maths for weighted multi-model deployments.

The class-alignment test is the one that matters: two members can order their
classes differently, and averaging column-against-column silently mixes one
class into another — no error, no crash, just a quietly worse model.
"""
import pytest

from app.services.ensemble_fusion import (
    fuse_classification,
    fuse_regression,
    normalise_weights,
)


class TestNormaliseWeights:
    def test_scales_to_one_keeping_ratios(self):
        assert normalise_weights([3, 1]) == pytest.approx([0.75, 0.25])

    def test_all_zero_falls_back_to_an_equal_split(self):
        # A deployment whose weights cancelled out should still predict.
        assert normalise_weights([0, 0, 0]) == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_negative_weights_are_clamped_not_subtracted(self):
        assert normalise_weights([-1, 1]) == pytest.approx([0.0, 1.0])


class TestFuseRegression:
    def test_weighted_average_per_row(self):
        fused = fuse_regression([[10.0, 20.0], [20.0, 40.0]], [0.5, 0.5])
        assert fused == pytest.approx([15.0, 30.0])

    def test_respects_unequal_weights(self):
        fused = fuse_regression([[0.0], [100.0]], [0.25, 0.75])
        assert fused == pytest.approx([75.0])

    def test_weights_need_not_be_pre_normalised(self):
        assert fuse_regression([[0.0], [100.0]], [1, 3]) == pytest.approx([75.0])

    def test_rejects_members_that_disagree_on_row_count(self):
        # Silently truncating here would return predictions for a different
        # number of rows than the caller sent.
        with pytest.raises(ValueError, match="行数不一致"):
            fuse_regression([[1.0, 2.0], [1.0]], [0.5, 0.5])


class TestFuseClassification:
    def test_aligns_classes_by_label_not_by_position(self):
        """The bug this module exists to prevent.

        Both members are certain the row is "dog", but they order their classes
        oppositely. Positional averaging would score cat and dog 0.5 each and
        pick whichever sorted first; label-keyed averaging gives dog 1.0.
        """
        members = [
            {"class_labels": ["cat", "dog"], "predictions": ["dog"], "probabilities": [[0.0, 1.0]]},
            {"class_labels": ["dog", "cat"], "predictions": ["dog"], "probabilities": [[1.0, 0.0]]},
        ]
        fused = fuse_classification(members, [0.5, 0.5])
        assert fused["predictions"] == ["dog"]
        assert fused["probabilities"][0]["dog"] == pytest.approx(1.0)
        assert fused["probabilities"][0]["cat"] == pytest.approx(0.0)

    def test_weights_shift_the_winner(self):
        members = [
            {"class_labels": ["a", "b"], "predictions": ["a"], "probabilities": [[0.9, 0.1]]},
            {"class_labels": ["a", "b"], "predictions": ["b"], "probabilities": [[0.2, 0.8]]},
        ]
        assert fuse_classification(members, [0.9, 0.1])["predictions"] == ["a"]
        assert fuse_classification(members, [0.1, 0.9])["predictions"] == ["b"]

    def test_keeps_a_class_only_one_member_knows(self):
        # A member that never saw a rare class must not erase it from the output.
        members = [
            {"class_labels": ["a", "b"], "predictions": ["a"], "probabilities": [[0.6, 0.4]]},
            {"class_labels": ["a", "b", "rare"], "predictions": ["rare"],
             "probabilities": [[0.1, 0.1, 0.8]]},
        ]
        fused = fuse_classification(members, [0.5, 0.5])
        assert "rare" in fused["class_labels"]
        assert fused["probabilities"][0]["rare"] > 0

    def test_a_member_without_probabilities_votes_with_its_hard_prediction(self):
        # Dropping it would silently change the weighting the caller configured.
        members = [
            {"class_labels": ["a", "b"], "predictions": ["a"], "probabilities": [[0.51, 0.49]]},
            {"class_labels": ["a", "b"], "predictions": ["b"], "probabilities": None},
        ]
        fused = fuse_classification(members, [0.3, 0.7])
        assert fused["predictions"] == ["b"]

    def test_probabilities_sum_to_one_per_row(self):
        members = [
            {"class_labels": ["a", "b"], "predictions": ["a"], "probabilities": [[0.7, 0.3]]},
            {"class_labels": ["a", "b"], "predictions": ["b"], "probabilities": [[0.4, 0.6]]},
        ]
        fused = fuse_classification(members, [0.5, 0.5])
        assert sum(fused["probabilities"][0].values()) == pytest.approx(1.0)

    def test_rejects_members_that_disagree_on_row_count(self):
        members = [
            {"class_labels": ["a"], "predictions": ["a", "a"], "probabilities": [[1.0], [1.0]]},
            {"class_labels": ["a"], "predictions": ["a"], "probabilities": [[1.0]]},
        ]
        with pytest.raises(ValueError, match="行数不一致"):
            fuse_classification(members, [0.5, 0.5])
