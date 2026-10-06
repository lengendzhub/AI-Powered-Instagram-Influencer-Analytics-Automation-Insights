# Data Schema — Instagram Influencer Analytics

## Profile Fields

| Field             | Type     | Source     | Example                  | Notes                           |
|-------------------|----------|------------|--------------------------|---------------------------------|
| handle            | str      | Input list | `fitnessguru_in`         | Unique identifier               |
| display_name      | str      | Profile    | `Fitness Guru India`     |                                 |
| follower_count    | int      | Profile    | `45230`                  |                                 |
| following_count   | int      | Profile    | `892`                    |                                 |
| post_count        | int      | Profile    | `312`                    | Total posts on profile          |
| bio               | str      | Profile    | `Certified trainer 🏋️`  | Full bio text                   |
| external_url      | str/null | Profile    | `https://fitguru.com`    | Link in bio                     |
| business_category | str/null | Profile    | `Fitness`                | Instagram's category label      |
| is_private        | bool     | Profile    | `false`                  | Skip if true                    |
| is_verified       | bool     | Profile    | `true`                   |                                 |
| niche_hint        | str      | Input list | `Fitness`                | From handle discovery           |
| scraped_at        | datetime | System     | `2026-10-03T14:22:00`    | UTC timestamp                   |

## Post Fields (last 10 posts per profile)

| Field      | Type     | Source | Example                       | Notes                     |
|------------|----------|--------|-------------------------------|---------------------------|
| post_id    | str      | Post   | `C8xK2...`                    | Instagram's shortcode     |
| timestamp  | datetime | Post   | `2026-09-15T10:30:00`         | UTC                       |
| likes      | int/null | Post   | `1234`                        | null if hidden by user    |
| comments   | int      | Post   | `56`                          |                           |
| caption    | str      | Post   | `Morning workout routine...`  | Full caption text         |
| hashtags   | list     | Post   | `["fitness", "gym", "india"]` | Extracted from caption    |
| media_type | str      | Post   | `image`/`video`/`carousel`    |                           |

## Derived Fields (Phase 1 — Day 5)

| Field               | Type       | Formula / Logic                                                     |
|----------------------|------------|---------------------------------------------------------------------|
| engagement_rate      | float/NaN  | mean( (likes + comments) / followers × 100 ) over valid posts       |
| engagement_rate_med  | float/NaN  | median of per-post engagement rates                                 |
| posts_per_month      | float      | num_posts / (date_range_days / 30) from last 10 post timestamps     |
| contact_email        | str        | Regex extraction from bio; empty string if none                     |
| top_hashtags         | str        | Top 5 most frequent hashtags, comma-separated                       |
| automation_flag      | int (0/1)  | 1 if any automation signal detected, else 0                         |
| automation_evidence  | str        | Semicolon-separated evidence strings                                |
| follower_tier        | str        | `nano` (<10K) / `micro` (10K–100K) / `mid` (100K–1M)               |

## Engineered Features (Phase 2 — Day 9)

| Field              | Type  | Description                                          |
|--------------------|-------|------------------------------------------------------|
| hashtag_diversity  | float | unique_hashtags / total_hashtags                     |
| hashtag_entropy    | float | Shannon entropy of hashtag distribution              |
| pct_image          | float | Share of image posts (0–1)                           |
| pct_video          | float | Share of video posts (0–1)                           |
| pct_carousel       | float | Share of carousel posts (0–1)                        |
| avg_caption_length | float | Mean character count of captions                     |
| avg_emoji_count    | float | Mean emoji characters per caption                    |
| has_cta            | int   | 1 if any caption contains CTA phrases                |
| has_link           | int   | 1 if any caption contains a URL                      |
| ff_ratio           | float | follower_count / following_count                     |
| posting_hour_std   | float | Std deviation of posting hour (regularity signal)    |

## Ranking Fields (Phase 2 — Day 12)

| Field             | Type  | Description                                      |
|-------------------|-------|--------------------------------------------------|
| influence_score   | float | Composite weighted score (0–1)                   |
| rank_overall      | int   | Overall ranking position                         |
| rank_in_category  | int   | Ranking position within niche                    |
| predicted_niche   | str   | NLP-predicted niche label                        |
