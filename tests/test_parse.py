"""
test_parse.py — Unit tests for the parsing functions.

Tests parsing logic using saved sample data (no Instaloader dependency needed).
Run with: python -m pytest tests/test_parse.py -v
"""

import pytest

from src.collect.parse import (
    compute_posting_regularity,
    detect_automation_signals,
    determine_media_type,
    extract_email,
    extract_hashtags,
    parse_profile_from_dict,
)


# ---------------------------------------------------------------------------
# Test extract_hashtags
# ---------------------------------------------------------------------------

class TestExtractHashtags:
    def test_basic_hashtags(self):
        text = "Love this #Fitness journey! #GymLife"
        result = extract_hashtags(text)
        assert result == ["fitness", "gymlife"]

    def test_no_hashtags(self):
        result = extract_hashtags("No hashtags here")
        assert result == []

    def test_empty_string(self):
        result = extract_hashtags("")
        assert result == []

    def test_none_input(self):
        result = extract_hashtags(None)
        assert result == []

    def test_multiple_hashtags(self):
        text = "#tech #review #gadget #india #2026"
        result = extract_hashtags(text)
        assert len(result) == 5
        assert "tech" in result
        assert "2026" in result

    def test_hashtags_with_underscores(self):
        text = "#fitness_motivation #gym_life"
        result = extract_hashtags(text)
        assert "fitness_motivation" in result
        assert "gym_life" in result


# ---------------------------------------------------------------------------
# Test extract_email
# ---------------------------------------------------------------------------

class TestExtractEmail:
    def test_basic_email(self):
        text = "DM or email me at hello@fitness.com for collabs"
        result = extract_email(text)
        assert result == "hello@fitness.com"

    def test_no_email(self):
        result = extract_email("No email here")
        assert result is None

    def test_empty_string(self):
        result = extract_email("")
        assert result is None

    def test_none_input(self):
        result = extract_email(None)
        assert result is None

    def test_complex_email(self):
        text = "Business: john.doe+work@company.co.in"
        result = extract_email(text)
        assert "john.doe" in result

    def test_multiple_emails_returns_first(self):
        text = "first@example.com or second@example.com"
        result = extract_email(text)
        assert result == "first@example.com"


# ---------------------------------------------------------------------------
# Test detect_automation_signals
# ---------------------------------------------------------------------------

class TestDetectAutomation:
    def test_linktree_in_bio(self):
        flag, evidence = detect_automation_signals("📧 linktr.ee/mypage | fitness coach", [])
        assert flag is True
        assert any("linktr.ee" in e for e in evidence)

    def test_no_signals(self):
        flag, evidence = detect_automation_signals("Just a fitness coach", ["Great workout!"])
        assert flag is False
        assert evidence == []

    def test_scheduler_in_bio(self):
        flag, evidence = detect_automation_signals("Managed by Buffer | Travel blogger", [])
        assert flag is True
        assert any("buffer" in e for e in evidence)

    def test_dm_automation_in_caption(self):
        flag, evidence = detect_automation_signals("", ["Comment GUIDE to get the free ebook!"])
        assert flag is True

    def test_multiple_signals(self):
        flag, evidence = detect_automation_signals(
            "linktr.ee/me | powered by Hootsuite",
            ["Comment GUIDE to get it!"],
        )
        assert flag is True
        assert len(evidence) >= 2

    def test_empty_inputs(self):
        flag, evidence = detect_automation_signals("", [])
        assert flag is False


# ---------------------------------------------------------------------------
# Test compute_posting_regularity
# ---------------------------------------------------------------------------

class TestPostingRegularity:
    def test_regular_posting(self):
        # Posts always at ~10 AM
        timestamps = [
            "2026-01-01T10:00:00",
            "2026-01-02T10:05:00",
            "2026-01-03T10:10:00",
            "2026-01-04T09:55:00",
        ]
        result = compute_posting_regularity(timestamps)
        assert result is not None
        assert result < 1.0  # Very regular

    def test_irregular_posting(self):
        # Posts at random times
        timestamps = [
            "2026-01-01T06:00:00",
            "2026-01-02T14:00:00",
            "2026-01-03T22:00:00",
            "2026-01-04T10:00:00",
        ]
        result = compute_posting_regularity(timestamps)
        assert result is not None
        assert result > 3.0  # Irregular

    def test_insufficient_data(self):
        result = compute_posting_regularity(["2026-01-01T10:00:00"])
        assert result is None

    def test_empty_list(self):
        result = compute_posting_regularity([])
        assert result is None


# ---------------------------------------------------------------------------
# Test parse_profile_from_dict
# ---------------------------------------------------------------------------

class TestParseProfileFromDict:
    def test_basic_profile(self):
        data = {
            "handle": "  FitnessGuru  ",
            "display_name": "Fitness Guru",
            "follower_count": "45000",
            "following_count": "892",
            "post_count": "312",
            "bio": "Certified trainer 🏋️",
            "external_url": "https://fitguru.com",
            "business_category": "Fitness",
            "is_private": False,
            "is_verified": True,
        }
        result = parse_profile_from_dict(data)

        assert result["handle"] == "fitnessguru"  # lowered and stripped
        assert result["follower_count"] == 45000
        assert result["is_verified"] is True
        assert result["is_private"] is False

    def test_missing_fields(self):
        data = {"handle": "minimal"}
        result = parse_profile_from_dict(data)

        assert result["handle"] == "minimal"
        assert result["follower_count"] == 0
        assert result["bio"] == ""
        assert result["is_private"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
