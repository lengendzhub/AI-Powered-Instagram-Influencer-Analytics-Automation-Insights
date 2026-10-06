"""
parse.py — Parsing functions for Instagram profile and post data.

Converts raw Instaloader objects into clean dictionaries ready for JSONL storage.
All parsing is isolated here so it can be unit-tested with saved sample JSON.
"""

import re
from datetime import datetime, timezone
from typing import Any, Optional


def parse_profile(profile: Any) -> dict:
    """
    Extract structured metadata from an Instaloader Profile object.

    Args:
        profile: An instaloader.Profile instance.

    Returns:
        A dict with profile fields matching schema.md.
    """
    bio = profile.biography or ""

    return {
        "handle": profile.username.lower(),
        "display_name": profile.full_name or "",
        "follower_count": profile.followers,
        "following_count": profile.followees,
        "post_count": profile.mediacount,
        "bio": bio,
        "external_url": profile.external_url or None,
        "business_category": getattr(profile, "business_category_name", None),
        "is_private": profile.is_private,
        "is_verified": profile.is_verified,
    }


def parse_post(post: Any) -> dict:
    """
    Extract structured metadata from an Instaloader Post object.

    Args:
        post: An instaloader.Post instance.

    Returns:
        A dict with post fields matching schema.md.
    """
    # Extract hashtags from caption
    caption = post.caption or ""
    hashtags = extract_hashtags(caption)

    # Determine media type
    media_type = determine_media_type(post)

    # Likes may be hidden (returns None or raises)
    likes = safe_get_likes(post)

    return {
        "post_id": post.shortcode,
        "timestamp": post.date_utc.isoformat() if post.date_utc else None,
        "likes": likes,
        "comments": post.comments if hasattr(post, "comments") else 0,
        "caption": caption,
        "hashtags": hashtags,
        "media_type": media_type,
    }


def extract_hashtags(text: str) -> list[str]:
    """
    Extract hashtags from a text string.

    Args:
        text: Caption or bio text.

    Returns:
        List of hashtag strings (without the # symbol), lowercased.

    Examples:
        >>> extract_hashtags("Love this #Fitness journey! #GymLife")
        ['fitness', 'gymlife']
        >>> extract_hashtags("No hashtags here")
        []
    """
    if not text:
        return []
    pattern = r"#(\w+)"
    return [tag.lower() for tag in re.findall(pattern, text)]


def extract_email(text: str) -> Optional[str]:
    """
    Extract the first email address from text using regex.

    Args:
        text: Bio or caption text.

    Returns:
        The first email found, or None.

    Examples:
        >>> extract_email("DM or email me at hello@fitness.com for collabs")
        'hello@fitness.com'
        >>> extract_email("No email here")
    """
    if not text:
        return None
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    return match.group(0) if match else None


def determine_media_type(post: Any) -> str:
    """
    Determine the media type of an Instaloader Post.

    Returns one of: 'image', 'video', 'carousel'.
    """
    if hasattr(post, "typename"):
        typename = post.typename.lower()
        if "sidecar" in typename or "carousel" in typename:
            return "carousel"
        if "video" in typename:
            return "video"
    if getattr(post, "is_video", False):
        return "video"
    return "image"


def safe_get_likes(post: Any) -> Optional[int]:
    """
    Safely get the like count from a post.

    Instagram allows users to hide like counts. In that case,
    return None rather than raising an error.

    Returns:
        Like count as int, or None if hidden/unavailable.
    """
    try:
        likes = post.likes
        if likes is None or likes < 0:
            return None
        return likes
    except Exception:
        return None


def detect_automation_signals(bio: str, captions: list[str]) -> tuple[bool, list[str]]:
    """
    Detect automation tool usage from bio and caption text.

    This is a HEURISTIC — it detects proxies for automation, not confirmed usage.
    Returns (flag, evidence_list).

    Signals checked:
      - Scheduler tools: Later, Buffer, Hootsuite, Planoly
      - Link-in-bio: Linktree, linktr.ee, Beacons, bio.link
      - DM automation: Manychat, "comment GUIDE", "comment below to get"
      - CTA automation phrases: "DM me [keyword] to get"

    Examples:
        >>> detect_automation_signals("📧 linktr.ee/mypage | fitness coach", [])
        (True, ['link-in-bio: linktr.ee'])
        >>> detect_automation_signals("Just a fitness coach", ["Great workout!"])
        (False, [])
    """
    evidence: list[str] = []
    all_text = (bio + " " + " ".join(captions)).lower()

    # Scheduler tools
    schedulers = ["later", "buffer", "hootsuite", "planoly", "sprout social"]
    for tool in schedulers:
        if tool in all_text:
            evidence.append(f"scheduler: {tool}")

    # Link-in-bio tools
    linkinbio = [
        "linktree", "linktr.ee", "beacons.ai", "bio.link",
        "lnk.bio", "campsite.bio", "tap.bio",
    ]
    for tool in linkinbio:
        if tool in all_text:
            evidence.append(f"link-in-bio: {tool}")

    # DM automation patterns
    dm_patterns = [
        r"comment\s+\w+\s+to\s+get",
        r"comment\s+\"\w+\"",
        r"dm\s+me\s+\w+\s+to\s+get",
        r"manychat",
        r"chatbot",
        r"auto.?reply",
        r"auto.?dm",
    ]
    for pattern in dm_patterns:
        if re.search(pattern, all_text):
            evidence.append(f"dm-automation: matched '{pattern}'")

    flag = len(evidence) > 0
    return flag, evidence


def compute_posting_regularity(timestamps: list[str]) -> Optional[float]:
    """
    Compute the standard deviation of posting hours.

    Lower std = more regular posting schedule (potential automation signal).

    Args:
        timestamps: List of ISO-format timestamp strings.

    Returns:
        Standard deviation of posting hour (0-23), or None if insufficient data.

    Examples:
        >>> # Posts always at 10 AM
        >>> compute_posting_regularity(["2026-01-01T10:00:00", "2026-01-02T10:05:00"])
        ... # Returns ~0.0 (very regular)
    """
    if not timestamps or len(timestamps) < 3:
        return None

    hours = []
    for ts in timestamps:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            hours.append(dt.hour + dt.minute / 60.0)
        except (ValueError, AttributeError):
            continue

    if len(hours) < 3:
        return None

    import statistics
    return statistics.stdev(hours)


def parse_profile_from_dict(data: dict) -> dict:
    """
    Parse a profile from a raw dictionary (e.g., from saved JSON fixture).

    Useful for unit testing without needing actual Instaloader objects.

    Args:
        data: Dictionary with raw profile fields.

    Returns:
        Cleaned profile dictionary.
    """
    return {
        "handle": data.get("handle", "").lower().strip(),
        "display_name": data.get("display_name", ""),
        "follower_count": int(data.get("follower_count", 0)),
        "following_count": int(data.get("following_count", 0)),
        "post_count": int(data.get("post_count", 0)),
        "bio": data.get("bio", ""),
        "external_url": data.get("external_url"),
        "business_category": data.get("business_category"),
        "is_private": bool(data.get("is_private", False)),
        "is_verified": bool(data.get("is_verified", False)),
    }
