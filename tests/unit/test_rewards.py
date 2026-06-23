"""Unit tests for reward functions."""

import pytest

from sdk.nura.rewards.resolution import ResolutionReward


class TestResolutionReward:
    def test_score_passthrough(self):
        reward = ResolutionReward()
        scores = reward.score(["p1", "p2"], ["c1", "c2"], [1.0, 0.0])
        assert scores == [1.0, 0.0]

    def test_score_clips_above_max(self):
        reward = ResolutionReward()
        scores = reward.score(["p"], ["c"], [1.5])
        assert scores == [1.0]

    def test_score_clips_below_min(self):
        reward = ResolutionReward()
        scores = reward.score(["p"], ["c"], [-0.5])
        assert scores == [0.0]

    def test_score_fractional_outcome(self):
        reward = ResolutionReward()
        scores = reward.score(["p"], ["c"], [0.72])
        assert scores == [pytest.approx(0.72)]

    def test_score_mismatched_lengths_raises(self):
        reward = ResolutionReward()
        with pytest.raises(ValueError, match="same length"):
            reward.score(["p1", "p2"], ["c1"], [1.0, 0.0])

    def test_validate_returns_true(self):
        assert ResolutionReward().validate() is True

    def test_custom_clip_bounds(self):
        reward = ResolutionReward(clip_min=0.1, clip_max=0.9)
        scores = reward.score(["p"], ["c"], [0.0])
        assert scores == [0.1]
        scores = reward.score(["p"], ["c"], [1.0])
        assert scores == [0.9]

    def test_invalid_clip_bounds_raises(self):
        with pytest.raises(ValueError):
            ResolutionReward(clip_min=1.0, clip_max=0.0)

    def test_empty_batch(self):
        reward = ResolutionReward()
        assert reward.score([], [], []) == []
