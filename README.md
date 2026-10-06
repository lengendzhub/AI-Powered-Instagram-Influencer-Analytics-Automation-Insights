# Instagram Influencer Analytics 🏆

> SR NEXT Self-Guided Data Project — Analyzing 2,000+ Instagram influencers across 5 niches with ML-powered engagement prediction, automation detection, and composite ranking.

## 🚀 Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/YOUR_USERNAME/sr-next-influencer.git
cd sr-next-influencer
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Mac/Linux
pip install -r requirements.txt

# 2. Collect data (Phase 1)
python -m src.collect.collect --config config.yaml

# 3. Compute Phase 1 features
python -m src.features.features_v1

# 4. Run Phase 2 pipeline (cleaning → features → NLP → ML → ranking)
python run_pipeline.py

# 5. Launch the dashboard
streamlit run app/dashboard.py
```

## 📂 Project Structure

```
sr-next-influencer/
├── data/
│   ├── raw/                  # Original scraped data
│   │   ├── handles.csv       # Input: 2,500+ Instagram handles
│   │   ├── profiles.jsonl    # Scraped profile + post data
│   │   └── errors.csv        # Scraping failure log
│   ├── interim/              # Partially processed
│   │   └── influencers_v1.csv
│   └── processed/            # Final clean datasets
│       ├── influencers_clean.csv
│       ├── features.csv
│       ├── ranked_influencers.csv
│       └── cleaning_log.csv
├── src/
│   ├── collect/              # Phase 1: Data collection
│   │   ├── discover.py       # Handle discovery helpers
│   │   ├── collect.py        # Resumable profile scraper
│   │   └── parse.py          # JSON parsing + feature extraction
│   ├── clean/                # Phase 2: Cleaning
│   │   └── clean.py          # 7-step cleaning pipeline
│   ├── features/             # Feature engineering
│   │   ├── features_v1.py    # Phase 1 derived features
│   │   └── features.py       # Phase 2 advanced features
│   ├── models/               # ML models
│   │   ├── nlp.py            # TF-IDF + niche classifier
│   │   └── models.py         # Engagement + automation models
│   └── rank/                 # Ranking
│       └── rank.py           # Composite scoring + sensitivity
├── app/
│   └── dashboard.py          # Streamlit + Plotly dashboard
├── reports/
│   ├── week1.md              # Week 1 report
│   ├── week2.md              # Week 2 report
│   └── sensitivity_analysis.md
├── models/                   # Saved model artifacts
├── tests/                    # Unit tests
├── config.yaml               # Central configuration
├── schema.md                 # Data schema documentation
├── run_pipeline.py           # Single pipeline entry point
└── requirements.txt
```

## 📊 Features

### Data Collection
- Resumable, rate-limited scraper using Instaloader
- Random delays (4–9s) with exponential backoff on rate limits
- Hard stop after 5 consecutive failures
- Automatic skip for private accounts and >1M follower accounts

### Analysis
- **Engagement metrics**: Mean/median engagement rate, posts per month
- **NLP classification**: TF-IDF + Logistic Regression/SVM niche classifier
- **Engagement prediction**: Linear Regression vs Random Forest vs Gradient Boosting
- **Automation detection**: Heuristic-based with ML classification
- **Composite ranking**: Weighted formula with sensitivity analysis

### Dashboard
- Interactive Streamlit + Plotly dashboard
- Filters by niche, follower tier, and top-N
- Engagement vs. followers scatter plot
- Category comparison box plots
- Hashtag frequency bar chart
- Automation adoption breakdown
- CSV download button

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

## 📝 Reports

- [Week 1 Report](reports/week1.md) — Data collection process, tools, and limitations
- [Week 2 Report](reports/week2.md) — Cleaning, analysis, and insights

## ⚠️ Ethics & Limitations

- Only public data is collected
- Automation labels are heuristic-based, not ground truth
- Niche labels come from discovery source, not manual annotation
- Hidden likes affect engagement rate coverage
- Dataset is for educational analysis only

## 📄 License

This project was built as part of the SR NEXT Self-Guided Data Project Program.
