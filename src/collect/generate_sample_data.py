"""
generate_sample_data.py — Generate realistic synthetic Instagram data for testing.

Creates:
  1. data/raw/handles.csv        — 2,500 handles across 5 niches
  2. data/raw/profiles.jsonl     — Simulated profile + post data

This allows the full pipeline to run end-to-end without actual scraping.
The synthetic data follows realistic distributions for follower counts,
engagement rates, posting patterns, and automation signals.

Usage:
    python -m src.collect.generate_sample_data [--count 2500]
"""

import argparse
import json
import logging
import random
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Realistic name / bio / caption generators
# ---------------------------------------------------------------------------

NICHE_BIOS: dict[str, list[str]] = {
    "Tech": [
        "Tech reviewer & gadget enthusiast 📱 | Honest reviews daily",
        "Software engineer sharing coding tips 💻 | Open source lover",
        "AI/ML researcher 🤖 | Breaking down complex tech simply",
        "Full stack developer 🚀 | Building the future, one line at a time",
        "Tech blogger & YouTuber | Unboxing, reviews, tutorials",
        "Startup founder | SaaS | Product builds & tech insights",
        "Data scientist by day, gamer by night 🎮 | Python advocate",
        "Cloud architect ☁️ | AWS | Azure | DevOps tips",
        "Mobile app developer 📲 | Flutter & React Native",
        "Cybersecurity analyst 🔐 | Ethical hacking | InfoSec tips",
    ],
    "Fashion": [
        "Fashion & lifestyle blogger 👗 | Sustainable style advocate",
        "Stylist & wardrobe consultant ✨ | DM for collabs",
        "Fashion designer | Handcrafted ethnic wear 🧵",
        "OOTD daily | Affordable fashion for everyone 💃",
        "Luxury fashion curator | Brand partnerships welcome",
        "Street style photographer 📸 | Capturing urban fashion",
        "Plus-size fashion advocate | Body positivity 💖",
        "Bridal fashion specialist 👰 | Wedding inspiration",
        "Thrift queen 👑 | Sustainable fashion on a budget",
        "Menswear enthusiast | Classic & contemporary style 🎩",
    ],
    "Fitness": [
        "Certified personal trainer 🏋️ | Transform your body",
        "Yoga instructor | Mindfulness & movement 🧘",
        "CrossFit athlete | Competition prep & daily WODs",
        "Nutritionist & meal prep expert 🥗 | Healthy eating made easy",
        "Marathon runner 🏃 | Training logs & race recaps",
        "Calisthenics coach | Bodyweight mastery 💪",
        "Fitness model | Contest prep | Science-based training",
        "Pilates instructor | Core strength & flexibility",
        "Boxing trainer 🥊 | Mental & physical toughness",
        "Gym owner | Helping 1000+ clients transform their lives",
    ],
    "Lifestyle": [
        "Living my best life ✨ | Mom of 2 | Coffee addict ☕",
        "Content creator | Daily vlogs | Life in Mumbai 🌆",
        "Minimalist living | Declutter your life & mind 🧹",
        "Foodie & home cook 🍳 | Recipe developer | Cookbook author",
        "Plant parent 🌿 | Interior decor | Cozy home vibes",
        "Book reviewer 📚 | 100 books a year challenge",
        "Digital nomad | Remote work tips & cafe reviews",
        "Self-improvement junkie | Productivity & habits 📝",
        "Pet mom 🐕 | Dog training tips & adoption advocacy",
        "DIY crafts & home projects | Making life beautiful 🎨",
    ],
    "Travel": [
        "Travel blogger ✈️ | 45 countries & counting 🌍",
        "Adventure photographer 📸 | Mountains, trails, & sunsets",
        "Budget travel expert | See the world without breaking the bank",
        "Solo female traveler | Safety tips & hidden gems 💎",
        "Luxury travel curator | Hotels, resorts, & experiences 🏖️",
        "Van life & road trips 🚐 | Full-time traveler",
        "Backpacker | Southeast Asia specialist | Hostel reviews",
        "Travel vlogger | Drone shots & destination guides 🎥",
        "Cultural explorer | Food, festivals, & traditions 🎭",
        "Weekend getaway specialist | Day trips & staycations 🏕️",
    ],
}

NICHE_HASHTAGS: dict[str, list[str]] = {
    "Tech": [
        "tech", "technology", "gadgets", "coding", "programming", "python",
        "developer", "software", "startup", "ai", "machinelearning", "devops",
        "webdev", "techreview", "innovation", "techie", "geek", "javascript",
        "opensource", "cloud", "cybersecurity", "iot", "datascience", "appdevelopment",
    ],
    "Fashion": [
        "fashion", "style", "ootd", "fashionblogger", "outfit", "streetstyle",
        "fashionista", "shopping", "designer", "luxury", "trend", "clothing",
        "accessories", "indianfashion", "ethnicwear", "sustainable", "vintage",
        "instafashion", "styleinspo", "lookoftheday", "fashionstyle", "wardrobe",
    ],
    "Fitness": [
        "fitness", "gym", "workout", "fitfam", "health", "exercise", "training",
        "yoga", "crossfit", "bodybuilding", "running", "fitnessmotivation",
        "healthylifestyle", "personaltrainer", "nutrition", "strength", "cardio",
        "gymlife", "fitlife", "transformation", "musclebuilding", "calisthenics",
    ],
    "Lifestyle": [
        "lifestyle", "life", "daily", "home", "food", "cooking", "recipe",
        "decor", "minimalism", "selfcare", "motivation", "productivity",
        "books", "reading", "plants", "diy", "craft", "coffee", "mindfulness",
        "homedecor", "organization", "wellness", "positivity", "inspiration",
    ],
    "Travel": [
        "travel", "wanderlust", "explore", "adventure", "nature", "photography",
        "travelblogger", "travelgram", "vacation", "roadtrip", "backpacking",
        "sunset", "mountains", "beach", "landscape", "destination", "passport",
        "travelphotography", "instatravel", "tourism", "solotravel", "hiking",
    ],
}

NICHE_CAPTIONS: dict[str, list[str]] = {
    "Tech": [
        "Just unboxed the new {product}! Initial impressions: incredible display, solid build quality. Full review dropping this week 📱 #techreview #gadgets",
        "5 Python tips that will make your code 10x cleaner 🐍 Save this for later! #coding #python #developer",
        "The future of AI is here. This model can generate images, write code, and reason about complex problems 🤯 #ai #machinelearning",
        "My home office setup tour 2026 🖥️ Spent months optimizing this workspace for productivity #desksetup #techie",
        "New tutorial: Building a REST API from scratch with FastAPI ⚡ Link in bio #webdev #tutorial #programming",
        "Hot take: This {product} is overpriced for what it offers. Here's why 💰 #techreview #honest",
        "Just hit 10K on GitHub! Thank you all for the support 🙏 #opensource #developer #milestone",
        "Cloud computing costs are out of control. Here are 7 ways to optimize your AWS bill ☁️ #cloud #devops",
        "This AI tool just saved me 20 hours of work this week. Thread on the best AI tools for developers 🧵 #ai #productivity",
        "Comparison: {product} vs {product}. Which one should you buy? Watch the full breakdown on my channel 📺 #versus #tech",
    ],
    "Fashion": [
        "Today's look 💕 This entire outfit cost under ₹2000! Affordable fashion is possible #ootd #affordablefashion",
        "Sustainable fashion isn't a trend, it's a responsibility 🌍 Wearing all thrifted pieces today #sustainable #thriftfind",
        "Wedding season styling tips 👰 5 ways to accessorize your lehenga this season #weddingfashion #indianwear",
        "New collection drop! Every piece is handcrafted by local artisans 🧵 DM for orders #handmade #designer",
        "Style hack: How to make one white shirt work for 7 different occasions 👔 Save this! #styletips #capsulewardrobe",
        "Festival fashion inspo 🪔 Mixing traditional with modern for the perfect Diwali look #festivalstyle #ethnic",
        "My favorite winter layering combinations ❄️ Cozy but make it chic #winterfashion #layering",
        "Accessories can completely transform an outfit. Swipe to see the before and after → #accessories #transformation",
        "Color of the season: sage green 🌿 3 ways to incorporate it into your wardrobe #colortrend #fashion2026",
        "Behind the scenes at fashion week! Such an honor to attend 📸 #fashionweek #backstage #runway",
    ],
    "Fitness": [
        "Morning workout done ✅ 5AM club never felt so good! Today's split: chest & triceps 💪 #gymlife #morningworkout",
        "Client transformation! 12 weeks of consistent training and nutrition. So proud 🏆 #transformation #coaching",
        "Form check: Common deadlift mistakes I see every day in the gym. Swipe to learn → #formcheck #deadlift",
        "Meal prep Sunday! 5 days of meals in 2 hours 🥗 Recipes in my latest blog post #mealprep #nutrition",
        "Just completed my first marathon in under 4 hours 🏃‍♂️ Training log and race recap on the blog #marathon #running",
        "You don't need a gym to build muscle. Full bodyweight workout you can do anywhere 🏠 #homeworkout #calisthenics",
        "Post-workout smoothie recipe that actually tastes good AND has 40g protein 🥤 Save this! #proteinshake #recipe",
        "Yoga flow for stress relief 🧘 15 minutes is all you need. Follow along video in stories #yoga #stressrelief",
        "Progressive overload is the key to growth. Here's how to track it properly 📊 #training #strength",
        "Rest day reminder: Recovery is where the magic happens 😴 Don't skip your rest days! #recovery #fitness",
    ],
    "Lifestyle": [
        "Sunday morning routine that changed my life ☀️ Journaling, coffee, and gratitude #morningroutine #selfcare",
        "Made this butter chicken from scratch and it's the BEST one yet 🍛 Recipe in stories #homecooking #recipe",
        "10 books that shaped my thinking this year 📚 Thread below ↓ #bookrecommendations #reading",
        "My apartment makeover is complete! Swipe to see the transformation 🏠 #homedecor #beforeandafter",
        "Adopted this sweet boy from the shelter today 🐕 Tips for first-time dog parents in my latest post #adoption",
        "Productivity system that actually works: time blocking + Pomodoro + weekly reviews 📝 #productivity #organization",
        "Plant tour! My indoor jungle has grown to 47 plants 🌿 Care tips for beginners in the carousel #plantparent",
        "Simple DIY project: turned old pallets into this gorgeous coffee table ☕ Full tutorial coming soon #diy #upcycle",
        "Coffee shop work session ☕ My top 5 cafes in Bangalore for remote work #remotework #cafereview",
        "Minimalism isn't about having less, it's about making room for more of what matters ✨ #minimalism #intentionalliving",
    ],
    "Travel": [
        "Sunrise at Hampi 🌅 This ancient city never fails to take my breath away #incredibleindia #hampi #sunrise",
        "Budget breakdown: 7 days in Goa for under ₹15,000 including flights! 🏖️ Full guide in bio #budgettravel #goa",
        "Hidden waterfall we found after a 3-hour trek through the Western Ghats 🌊 Worth every step #adventure #trekking",
        "Solo travel tip: Always share your itinerary with someone back home 📍 Safety first! #solotravel #traveltips",
        "This resort in Kerala is absolute paradise 🌴 Swipe for the infinity pool view #luxurytravel #kerala",
        "Road trip essentials: My packing list for a week-long drive through Ladakh 🚗 #roadtrip #ladakh",
        "Street food tour in Old Delhi! Everything we tried for under ₹500 🍛 Full video on YouTube #streetfood #delhi",
        "Camping under the stars in Spiti Valley ⛺ No WiFi, no problems #vanlife #spitivalley #camping",
        "Cultural experience: Attending a traditional dance performance in Rajasthan 💃 #culture #rajasthan",
        "Best time to visit Manali: October-November for clear skies and no crowds 🏔️ #traveltips #manali",
    ],
}

AUTOMATION_BIOS = [
    "📧 linktr.ee/{handle} | {niche} content",
    "DM 'GUIDE' for free ebook! 📖 | {niche} tips daily",
    "Managed by Buffer 📅 | {niche} creator",
    "beacons.ai/{handle} | Comment 'START' for free course",
    "Hootsuite certified 📊 | {niche} strategist",
    "bio.link/{handle} | Manychat automated DMs 🤖",
]

DISPLAY_NAME_PREFIXES = {
    "Tech": ["Tech", "Code", "Dev", "Cyber", "Digital", "Pixel", "Byte", "Data", "Cloud", "AI"],
    "Fashion": ["Style", "Chic", "Glam", "Vogue", "Trend", "Luxe", "Mode", "Couture", "Fash", "Drape"],
    "Fitness": ["Fit", "Gym", "Iron", "Flex", "Strong", "Power", "Muscle", "Active", "Core", "Lift"],
    "Lifestyle": ["Life", "Vibe", "Soul", "Zen", "Bloom", "Glow", "Joy", "Cozy", "Grace", "Rise"],
    "Travel": ["Wander", "Globe", "Atlas", "Roam", "Voyage", "Trek", "Drift", "Explore", "Nomad", "Trail"],
}

DISPLAY_NAME_SUFFIXES = [
    "Hub", "Daily", "World", "India", "Official", "Pro", "Studio",
    "Zone", "Guide", "Guru", "Expert", "Master", "Club", "Spot",
    "Tales", "Diaries", "Journal", "Chronicles", "Stories", "Vibes",
]

PRODUCTS = ["Galaxy S26", "iPhone 18", "Pixel 11", "OnePlus 14", "MacBook Pro M5",
            "iPad Air", "Surface Pro 12", "ThinkPad X2", "ROG Ally 2", "Steam Deck 2"]

BUSINESS_CATEGORIES = {
    "Tech": ["Technology", "Science & Technology", "Software", "Gaming", None],
    "Fashion": ["Fashion", "Clothing & Apparel", "Beauty & Personal Care", "Shopping", None],
    "Fitness": ["Fitness", "Health & Wellness", "Sports", "Personal Training", None],
    "Lifestyle": ["Lifestyle", "Food & Beverage", "Home & Garden", "Personal Blog", None],
    "Travel": ["Travel & Tourism", "Photography", "Adventure", "Hospitality", None],
}


def generate_handle(niche: str, index: int) -> str:
    """Generate a realistic Instagram handle."""
    prefixes = DISPLAY_NAME_PREFIXES[niche]
    prefix = random.choice(prefixes).lower()
    styles = [
        f"{prefix}_{random.choice(['with', 'by', 'of'])}_{random.choice(['raj', 'priya', 'amit', 'neha', 'arjun', 'sneha', 'vikram', 'ananya', 'rohan', 'kavya', 'aditya', 'maya', 'karan', 'sana', 'deepak'])}",
        f"{prefix}{random.choice(DISPLAY_NAME_SUFFIXES).lower()}{random.randint(1, 99)}",
        f"the{prefix}{random.choice(['guru', 'pro', 'master', 'king', 'queen'])}",
        f"{random.choice(['real', 'its', 'official', 'the', 'im'])}{prefix}{random.choice(['india', 'daily', 'hq'])}",
        f"{prefix}{''.join(random.choices(string.digits, k=random.randint(1, 3)))}",
    ]
    handle = random.choice(styles).replace(" ", "").lower()
    # Ensure uniqueness by appending index if needed
    return f"{handle}_{index}"


def generate_display_name(niche: str, handle: str) -> str:
    """Generate a display name from a handle."""
    prefix = random.choice(DISPLAY_NAME_PREFIXES[niche])
    suffix = random.choice(DISPLAY_NAME_SUFFIXES)
    styles = [
        f"{prefix} {suffix}",
        f"{prefix} {suffix} 🇮🇳",
        f"The {prefix} {suffix}",
        handle.replace("_", " ").title(),
    ]
    return random.choice(styles)


def generate_follower_count() -> int:
    """Generate realistic follower count (power-law distribution, <1M)."""
    # Most influencers are nano/micro; few are mid-tier
    tier = random.choices(
        ["nano", "micro", "mid"],
        weights=[0.55, 0.35, 0.10],
        k=1,
    )[0]

    if tier == "nano":
        return int(np.random.lognormal(mean=8.0, sigma=0.8))  # ~1K-10K
    elif tier == "micro":
        return int(np.random.lognormal(mean=10.2, sigma=0.6))  # ~10K-100K
    else:
        return int(np.random.lognormal(mean=11.8, sigma=0.4))  # ~100K-999K


def generate_posts(
    niche: str,
    follower_count: int,
    num_posts: int = 10,
    has_automation: bool = False,
) -> list[dict]:
    """Generate realistic post data."""
    posts = []
    niche_tags = NICHE_HASHTAGS[niche]
    niche_captions = NICHE_CAPTIONS[niche]

    # Determine posting regularity
    if has_automation:
        # Automated accounts post at regular times
        base_hour = random.randint(8, 20)
        hour_std = random.uniform(0.5, 1.5)
    else:
        base_hour = random.randint(6, 22)
        hour_std = random.uniform(2.0, 8.0)

    # Determine base engagement rate (inverse correlation with followers)
    if follower_count < 10_000:
        base_er = random.uniform(2.0, 8.0)
    elif follower_count < 100_000:
        base_er = random.uniform(1.0, 4.0)
    else:
        base_er = random.uniform(0.5, 2.5)

    # Generate posts going backwards in time
    now = datetime.now(timezone.utc)
    days_between = random.uniform(1.5, 7.0)  # days between posts

    for i in range(num_posts):
        post_date = now - timedelta(days=days_between * (i + 1) + random.uniform(-0.5, 0.5))
        hour = max(0, min(23, int(np.random.normal(base_hour, hour_std))))
        post_date = post_date.replace(hour=hour, minute=random.randint(0, 59))

        # Engagement for this post
        er_this = max(0, np.random.normal(base_er, base_er * 0.3))
        likes = max(0, int(follower_count * er_this / 100))
        comments = max(0, int(likes * random.uniform(0.02, 0.08)))

        # Sometimes likes are hidden (~15% of posts)
        likes_hidden = random.random() < 0.15

        # Caption
        caption = random.choice(niche_captions)
        for product in PRODUCTS:
            caption = caption.replace("{product}", product, 1)

        # Hashtags
        num_tags = random.randint(3, 12)
        hashtags = random.sample(niche_tags, min(num_tags, len(niche_tags)))
        # Add some cross-niche tags
        cross_tags = ["india", "instagram", "instagood", "photooftheday", "viral", "trending"]
        hashtags.extend(random.sample(cross_tags, random.randint(1, 3)))

        # Media type
        media_type = random.choices(
            ["image", "video", "carousel"],
            weights=[0.50, 0.30, 0.20],
            k=1,
        )[0]

        posts.append({
            "post_id": "".join(random.choices(string.ascii_letters + string.digits, k=11)),
            "timestamp": post_date.isoformat(),
            "likes": None if likes_hidden else likes,
            "comments": comments,
            "caption": caption + " " + " ".join(f"#{t}" for t in hashtags),
            "hashtags": hashtags,
            "media_type": media_type,
        })

    return posts


def generate_profile(niche: str, index: int, seed_offset: int = 0) -> dict:
    """Generate a single realistic profile record."""
    handle = generate_handle(niche, index)
    display_name = generate_display_name(niche, handle)
    follower_count = generate_follower_count()
    follower_count = min(follower_count, 999_000)  # Cap below 1M

    following_count = int(follower_count * random.uniform(0.01, 0.5))
    following_count = max(50, min(following_count, 7500))
    post_count = random.randint(30, 2000)

    # Automation (~20% of accounts)
    has_automation = random.random() < 0.20

    # Bio
    if has_automation:
        bio_template = random.choice(AUTOMATION_BIOS)
        bio = bio_template.format(handle=handle, niche=niche)
    else:
        bio = random.choice(NICHE_BIOS[niche])

    # Some profiles have email in bio (~30%)
    if random.random() < 0.30:
        email_user = handle.split("_")[0]
        bio += f" | 📧 {email_user}@gmail.com"

    # External URL (~40%)
    external_url = None
    if random.random() < 0.40:
        url_choices = [
            f"https://www.youtube.com/@{handle}",
            f"https://linktr.ee/{handle}",
            f"https://{handle.split('_')[0]}.com",
            f"https://bio.link/{handle}",
        ]
        external_url = random.choice(url_choices)

    # Business category (~60% have one)
    business_category = random.choice(BUSINESS_CATEGORIES[niche])

    # Verified (~5%)
    is_verified = random.random() < 0.05

    # Generate posts
    posts = generate_posts(niche, follower_count, num_posts=10, has_automation=has_automation)

    return {
        "handle": handle,
        "display_name": display_name,
        "follower_count": follower_count,
        "following_count": following_count,
        "post_count": post_count,
        "bio": bio,
        "external_url": external_url,
        "business_category": business_category,
        "is_private": False,
        "is_verified": is_verified,
        "niche_hint": niche,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "posts": posts,
        "posts_fetched": len(posts),
    }


def generate_dataset(
    count: int = 2500,
    config_path: str = "config.yaml",
    seed: int = 42,
) -> None:
    """
    Generate the full synthetic dataset.

    Creates handles.csv and profiles.jsonl with realistic distributions.
    """
    config = load_config(config_path)
    niches = config.get("niches", ["Tech", "Fashion", "Fitness", "Lifestyle", "Travel"])

    random.seed(seed)
    np.random.seed(seed)

    per_niche = count // len(niches)
    remainder = count % len(niches)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    handles_path = RAW_DIR / "handles.csv"
    profiles_path = RAW_DIR / "profiles.jsonl"

    logger.info("Generating %d synthetic profiles across %d niches...", count, len(niches))

    # Generate handles.csv
    all_handles = []
    all_profiles = []
    global_idx = 0

    for i, niche in enumerate(niches):
        n = per_niche + (1 if i < remainder else 0)
        logger.info("  %s: %d profiles", niche, n)

        for j in range(n):
            profile = generate_profile(niche, global_idx)
            all_handles.append(f"{profile['handle']},{niche}")
            all_profiles.append(profile)
            global_idx += 1

    # Shuffle to avoid niche-ordered data
    combined = list(zip(all_handles, all_profiles))
    random.shuffle(combined)
    all_handles, all_profiles = zip(*combined)

    # Write handles.csv
    with open(handles_path, "w", encoding="utf-8") as f:
        f.write("handle,niche_hint\n")
        for line in all_handles:
            f.write(line + "\n")
    logger.info("Saved %d handles to %s", len(all_handles), handles_path)

    # Write profiles.jsonl
    with open(profiles_path, "w", encoding="utf-8") as f:
        for profile in all_profiles:
            f.write(json.dumps(profile, ensure_ascii=False, default=str) + "\n")
    logger.info("Saved %d profiles to %s", len(all_profiles), profiles_path)

    # Print distribution summary
    from collections import Counter
    niche_dist = Counter(p["niche_hint"] for p in all_profiles)
    tier_dist = Counter()
    for p in all_profiles:
        fc = p["follower_count"]
        if fc < 10_000:
            tier_dist["nano"] += 1
        elif fc < 100_000:
            tier_dist["micro"] += 1
        else:
            tier_dist["mid"] += 1

    auto_count = sum(1 for p in all_profiles if any(
        kw in p["bio"].lower()
        for kw in ["linktr.ee", "buffer", "hootsuite", "manychat", "comment", "beacons"]
    ))

    logger.info("")
    logger.info("=" * 60)
    logger.info("  SYNTHETIC DATA GENERATED")
    logger.info("=" * 60)
    logger.info("  Total profiles: %d", len(all_profiles))
    logger.info("")
    logger.info("  Niche distribution:")
    for niche, cnt in sorted(niche_dist.items()):
        logger.info("    %s: %d", niche, cnt)
    logger.info("")
    logger.info("  Follower tier distribution:")
    for tier, cnt in sorted(tier_dist.items()):
        logger.info("    %s: %d", tier, cnt)
    logger.info("")
    logger.info("  Automation signals: %d (%.1f%%)", auto_count, auto_count / len(all_profiles) * 100)
    logger.info("")
    logger.info("  Files created:")
    logger.info("    %s", handles_path)
    logger.info("    %s", profiles_path)
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Instagram data")
    parser.add_argument("--count", type=int, default=2500, help="Number of profiles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    generate_dataset(count=args.count, config_path=args.config, seed=args.seed)


if __name__ == "__main__":
    main()
