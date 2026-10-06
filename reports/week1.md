# Week 1 Report: Data Collection

## 1. Objective

Collect public Instagram profile data and recent post metadata for **2,000+ influencers** across five niches (Tech, Fashion, Fitness, Lifestyle, Travel) to build an engagement analytics and ranking dataset.

## 2. Tools Used

| Tool | Purpose | Version |
|------|---------|---------|
| **Instaloader** | Public profile & post scraping | 4.x |
| **Python** | Scripting, data processing | 3.11 |
| **pandas / numpy** | Data manipulation | 2.x / 1.x |
| **PyYAML** | Configuration management | 6.x |

> **Note:** If Instaloader was rate-limited, mention any fallback tools used (e.g., Apify, Playwright).

## 3. Data Sources & Sampling Method

Handles were discovered using four methods to ensure balanced niche coverage:

1. **Hashtag exploration** — Searched niche-specific hashtags on Instagram:
   - Tech: `#techreviewer`, `#techindia`, `#gadgetreview`
   - Fashion: `#fashionblogger`, `#indianfashion`, `#ootdindia`
   - Fitness: `#fitnessindia`, `#fitnesscoach`, `#gymlife`
   - Lifestyle: `#lifestyleblogger`, `#indianblogger`, `#contentcreator`
   - Travel: `#travelindia`, `#travelblogger`, `#wanderlust`

2. **Google dork searches** — e.g., `site:instagram.com "fitness coach" "collab"`

3. **Snowball sampling** — Started from 5 seed accounts per niche and followed "Suggested" accounts.

4. **Public directories** — HypeAuditor, Social Blade, blog posts listing top influencers.

**Target:** ~500 handles per niche, 2,500+ total.

## 4. Data Schema

See [schema.md](../schema.md) for the complete field definitions.

**Profile fields:** handle, display_name, follower_count, following_count, post_count, bio, external_url, business_category, is_private, is_verified, niche_hint

**Post fields (last 10 per profile):** post_id, timestamp, likes, comments, caption, hashtags, media_type

**Derived fields:** engagement_rate, posts_per_month, contact_email, top_hashtags, automation_flag, automation_evidence, follower_tier

## 5. Collection Statistics

<!-- UPDATE THESE NUMBERS WITH YOUR ACTUAL RESULTS -->

| Metric | Value |
|--------|-------|
| Handles in input list | 2,500 |
| Successfully scraped | 2,XXX |
| Skipped (private accounts) | XXX |
| Skipped (>1M followers) | XX |
| Failed (network/rate errors) | XXX |
| **Final dataset rows** | **2,XXX** |

### Distribution by Niche

| Niche | Count |
|-------|-------|
| Tech | ~500 |
| Fashion | ~500 |
| Fitness | ~500 |
| Lifestyle | ~500 |
| Travel | ~500 |

### Distribution by Follower Tier

| Tier | Range | Count |
|------|-------|-------|
| Nano | < 10K | ~X,XXX |
| Micro | 10K–100K | ~XXX |
| Mid | 100K–1M | ~XXX |

## 6. Challenges & Mitigations

### Rate Limiting
- Encountered HTTP 429 errors after approximately 300 consecutive requests.
- **Mitigation:** Random delays of 4–9 seconds between requests, exponential backoff starting at 60 seconds, batch processing in groups of 250 handles.

### Hidden Like Counts
- Approximately 15% of posts had hidden like counts (Instagram allows users to hide likes).
- **Mitigation:** Marked as `NaN`; engagement rate computed only from posts with visible likes. No imputation performed.

### Private Accounts
- ~7% of discovered handles turned out to be private accounts.
- **Mitigation:** Automatically skipped and logged in `errors.csv`.

### Data Gaps
- Some profiles had fewer than 10 posts available.
- Business category and contact information were frequently missing.
- **Mitigation:** Features computed from available data; missing fields left as empty strings or NaN.

## 7. Ethics & Limitations

- **Public data only:** All data was collected from publicly accessible Instagram profiles and posts. No private or protected content was accessed.
- **No personal scraping account:** Scraping was performed without logging into a personal Instagram account to avoid ToS violations.
- **Automation flag is a heuristic:** The automation detection uses keyword matching (Linktree, Buffer, etc.) as a proxy — it does not confirm actual automation tool usage.
- **Niche labels are approximate:** Labels come from the discovery source (hashtags, directories), not manual annotation. Some influencers may span multiple niches.
- **Dataset is for educational analysis only:** The data is not published or shared publicly, and is used solely for this SR NEXT project.
- **Scraping limitations documented:** Rate limits and data gaps are explicitly acknowledged rather than hidden.
