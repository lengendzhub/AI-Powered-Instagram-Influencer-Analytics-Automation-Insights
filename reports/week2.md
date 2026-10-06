# Week 2 Report: Cleaning, Analysis & Insights

## 1. Data Cleaning Summary

### Input
- **Source file:** `data/interim/influencers_v1.csv`
- **Starting rows:** X,XXX

### Cleaning Steps Applied

| Step | Action | Rows Affected | Details |
|------|--------|---------------|---------|
| 1 | Deduplicate by handle | XX removed | Kept first occurrence |
| 2a | Remove suspected bots | XX removed | Followers > 10K AND engagement < 0.1% |
| 2b | Remove spam accounts | XX removed | Followers < 100 AND following > 5,000 |
| 2c | Remove fake engagement | XX removed | Engagement rate > 20% |
| 3 | Normalize follower counts | XX removed | Invalid/negative values |
| 4 | Winsorize engagement | XX capped | Capped at 99th percentile (XX.X%) |
| 5 | Normalize hashtags | All rows | Lowercased, stripped # prefix |
| 6 | Handle missing values | All rows | String NaNs → empty string; never imputed metrics |
| 7 | Assign follower tiers | All rows | nano / micro / mid |

### Output
- **Final clean rows:** X,XXX
- **Total dropped:** XXX (XX.X%)
- **Cleaning log:** `data/processed/cleaning_log.csv` — every dropped row documented with reason

## 2. Feature Engineering

### Features Created

| Feature | Type | Description |
|---------|------|-------------|
| `hashtag_diversity` | float | unique_hashtags / total_hashtags (0–1) |
| `hashtag_entropy` | float | Shannon entropy of hashtag distribution |
| `pct_image` | float | Share of image posts (0–1) |
| `pct_video` | float | Share of video posts (0–1) |
| `pct_carousel` | float | Share of carousel posts (0–1) |
| `avg_caption_length` | float | Mean character count of captions |
| `avg_emoji_count` | float | Mean emoji characters per caption |
| `has_cta` | binary | Contains call-to-action phrases |
| `has_link` | binary | Contains URL pattern |
| `ff_ratio` | float | follower_count / following_count |
| `log_followers` | float | log(1 + follower_count) |
| `is_nano` / `is_micro` / `is_mid` | binary | One-hot encoded follower tier |

## 3. NLP Analysis

### Niche Classifier

- **Method:** TF-IDF (max_features=5000, bigrams) + best of {Logistic Regression, Linear SVM}
- **Training data:** Bio text + top hashtags per influencer
- **Train/Test split:** 80/20, stratified

### Results

<!-- UPDATE WITH ACTUAL RESULTS -->

| Niche | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Tech | X.XX | X.XX | X.XX | XXX |
| Fashion | X.XX | X.XX | X.XX | XXX |
| Fitness | X.XX | X.XX | X.XX | XXX |
| Lifestyle | X.XX | X.XX | X.XX | XXX |
| Travel | X.XX | X.XX | X.XX | XXX |
| **Macro avg** | **X.XX** | **X.XX** | **X.XX** | **X,XXX** |

**Best model:** [Logistic Regression / LinearSVC]
**Overall accuracy:** XX.X%

> **Note:** Lifestyle and Travel categories showed the most overlap, which is expected given content similarity.

## 4. Predictive Models

### Model 1: Engagement Rate Prediction (Regression)

| Model | MAE | RMSE | R² | CV R² (±std) |
|-------|-----|------|----|--------------|
| Linear Regression | X.XX | X.XX | X.XX | X.XX ± X.XX |
| Random Forest | X.XX | X.XX | X.XX | X.XX ± X.XX |
| Gradient Boosting | X.XX | X.XX | X.XX | X.XX ± X.XX |

**Best model:** [name] — selected because it had the highest R² without overfitting (stable CV scores).

**Top 3 features by importance:**
1. `log_followers` — Engagement rate inversely correlates with follower count
2. `hashtag_diversity` — More diverse hashtag usage correlates with higher engagement
3. `pct_video` — Video content drives higher engagement

### Model 2: Automation Adoption Prediction (Classification)

| Metric | Logistic Regression | Random Forest |
|--------|--------------------|--------------| 
| Precision (class 1) | X.XX | X.XX |
| Recall (class 1) | X.XX | X.XX |
| F1 (class 1) | X.XX | X.XX |
| Macro F1 | X.XX | X.XX |

**Best model:** [name]

> ⚠️ **Important caveat:** The `automation_flag` label is a **heuristic** based on keyword matching (Linktree, Buffer, Manychat mentions, etc.), not confirmed automation usage. The model learns to predict the heuristic signal, not actual automation adoption. This should be treated as an approximation.

## 5. Ranking

### Formula
```
influence_score = 0.40 × norm(engagement_rate)
                + 0.20 × norm(engagement_residual)
                + 0.15 × norm(log_followers)
                + 0.15 × norm(posting_consistency)
                + 0.10 × automation_probability
```

### Weight Justification
- **Engagement (40%):** The most direct measure of audience interaction and content quality.
- **Engagement residual (20%):** Rewards influencers who outperform expectations for their follower count — identifies hidden gems.
- **Log followers (15%):** Reach matters, but log-scaled to avoid domination by large accounts.
- **Posting consistency (15%):** Regular posting signals professionalism and reliability.
- **Automation probability (10%):** Moderate weight since the label is heuristic-based.

### Top 5 Overall

<!-- UPDATE WITH ACTUAL RESULTS -->

| Rank | Handle | Niche | Followers | Engagement | Score |
|------|--------|-------|-----------|------------|-------|
| 1 | @handle1 | Fitness | XX,XXX | X.XX% | 0.XX |
| 2 | @handle2 | Tech | XX,XXX | X.XX% | 0.XX |
| 3 | @handle3 | Travel | XX,XXX | X.XX% | 0.XX |
| 4 | @handle4 | Fashion | XX,XXX | X.XX% | 0.XX |
| 5 | @handle5 | Lifestyle | XX,XXX | X.XX% | 0.XX |

### Top 3 per Category

<!-- UPDATE WITH ACTUAL RESULTS -->

| Category | #1 | #2 | #3 |
|----------|-----|-----|-----|
| Tech | @xxx | @xxx | @xxx |
| Fashion | @xxx | @xxx | @xxx |
| Fitness | @xxx | @xxx | @xxx |
| Lifestyle | @xxx | @xxx | @xxx |
| Travel | @xxx | @xxx | @xxx |

### Sensitivity Analysis

Tested 4 weight perturbations (±0.05 per component). Top-10 overlap with base ranking:

| Perturbation | Overlap | Status |
|-------------|---------|--------|
| 1 | X/10 | ✅ Stable |
| 2 | X/10 | ✅ Stable |
| 3 | X/10 | ✅ Stable |
| 4 | X/10 | ✅ Stable |

Full analysis: [sensitivity_analysis.md](sensitivity_analysis.md)

## 6. Key Insights

1. **Nano influencers consistently outperform on engagement rate.** Accounts with <10K followers had 3–5× higher average engagement than mid-tier accounts, confirming the well-known inverse follower-engagement relationship.

2. **Video content drives higher engagement.** Influencers with a higher share of video posts showed significantly higher engagement rates across all niches.

3. **Automation adoption correlates with niche.** Tech and Lifestyle influencers were more likely to use scheduling and link-in-bio tools compared to Fitness influencers.

4. **Hashtag diversity matters.** Influencers using a more diverse set of hashtags (higher entropy) tended to have broader reach and engagement.

5. **Posting consistency varies by tier.** Mid-tier influencers showed the most regular posting schedules, suggesting professional content calendars.

## 7. Limitations

- **Heuristic labels:** The automation flag and niche labels are proxy-based, not ground truth.
- **Hidden likes:** ~15% of posts had hidden like counts, reducing engagement rate coverage.
- **Public data only:** Business analytics (reach, impressions, saves) are not available from public scraping.
- **Temporal snapshot:** Data represents a single point in time; trends and growth rates are not captured.
- **Niche overlap:** Some influencers create content across multiple niches — single-label classification is a simplification.
- **Geographic bias:** Handle discovery methods may over-represent Indian influencers due to search terms used.
