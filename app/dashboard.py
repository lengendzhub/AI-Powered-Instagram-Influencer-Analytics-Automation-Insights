"""
dashboard.py — Streamlit + Plotly interactive dashboard for influencer analytics.

Provides:
  1. Key metrics cards
  2. Sidebar filters (niche, tier, top-N)
  3. Top-N leaderboard table
  4. Engagement vs followers scatter plot
  5. Category comparison box plots
  6. Hashtag bar chart
  7. Automation adoption breakdown
  8. Download ranked CSV button

Run with: streamlit run app/dashboard.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Instagram Influencer Analytics",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 14px;
        opacity: 0.9;
        font-weight: 400;
    }
    .metric-card h1 {
        margin: 5px 0 0 0;
        font-size: 28px;
        font-weight: 700;
    }
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the ranked influencer dataset."""
    # Try multiple possible paths
    paths = [
        Path("data/processed/ranked_influencers.csv"),
        Path("../data/processed/ranked_influencers.csv"),
        Path("data/processed/features.csv"),
        Path("../data/processed/features.csv"),
    ]

    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            st.sidebar.success(f"Loaded {len(df)} influencers")
            return df

    st.error("❌ No data file found. Run the pipeline first!")
    st.stop()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # --- Header ---
    st.title("🏆 Instagram Influencer Analytics Dashboard")
    st.markdown("*Discover, analyze, and rank Instagram influencers across niches*")
    st.markdown("---")

    # --- Load data ---
    df = load_data()

    # --- Sidebar filters ---
    st.sidebar.title("🔍 Filters")

    # Niche filter
    niche_col = "predicted_niche" if "predicted_niche" in df.columns else "niche_hint"
    available_niches = sorted(df[niche_col].dropna().unique())
    selected_niches = st.sidebar.multiselect(
        "Category / Niche",
        options=available_niches,
        default=available_niches,
    )

    # Follower tier filter
    if "follower_tier" in df.columns:
        available_tiers = sorted(df["follower_tier"].dropna().unique())
        selected_tiers = st.sidebar.multiselect(
            "Follower Tier",
            options=available_tiers,
            default=available_tiers,
        )
    else:
        selected_tiers = None

    # Top-N slider
    top_n = st.sidebar.slider("Top-N Leaderboard", min_value=5, max_value=100, value=20, step=5)

    # Apply filters
    mask = df[niche_col].isin(selected_niches)
    if selected_tiers is not None and "follower_tier" in df.columns:
        mask &= df["follower_tier"].isin(selected_tiers)
    df_filtered = df[mask].copy()

    if df_filtered.empty:
        st.warning("No data matches the selected filters.")
        return

    # --- Key Metrics Cards ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total Influencers", f"{len(df_filtered):,}")

    with col2:
        avg_er = df_filtered["engagement_rate"].mean()
        st.metric("💡 Avg Engagement Rate", f"{avg_er:.2f}%" if not pd.isna(avg_er) else "N/A")

    with col3:
        if "influence_score" in df_filtered.columns and not df_filtered["influence_score"].isna().all():
            top_performer = df_filtered.loc[df_filtered["influence_score"].idxmax()]
            st.metric("🥇 Top Performer", f"@{top_performer['handle']}")
        else:
            st.metric("🥇 Top Performer", "N/A")

    with col4:
        if "automation_flag" in df_filtered.columns:
            auto_pct = df_filtered["automation_flag"].mean() * 100
            st.metric("🤖 Using Automation", f"{auto_pct:.1f}%")
        else:
            st.metric("🤖 Using Automation", "N/A")

    st.markdown("---")

    # --- Top-N Leaderboard ---
    st.subheader(f"🏅 Top {top_n} Influencers")

    leaderboard_cols = ["handle", niche_col, "follower_count", "engagement_rate"]
    if "influence_score" in df_filtered.columns:
        sort_col = "influence_score"
        leaderboard_cols.append("influence_score")
    else:
        sort_col = "engagement_rate"
    if "rank_overall" in df_filtered.columns:
        leaderboard_cols.insert(0, "rank_overall")

    available_cols = [c for c in leaderboard_cols if c in df_filtered.columns]
    leaderboard = df_filtered.nlargest(top_n, sort_col)[available_cols]

    st.dataframe(
        leaderboard.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # --- Charts row 1: Scatter + Box plots ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📈 Engagement vs Followers")

        scatter_df = df_filtered.dropna(subset=["follower_count", "engagement_rate"]).copy()
        if not scatter_df.empty:
            scatter_df["log_followers_plot"] = np.log10(scatter_df["follower_count"].clip(lower=1))

            fig = px.scatter(
                scatter_df,
                x="log_followers_plot",
                y="engagement_rate",
                color=niche_col,
                size="influence_score" if "influence_score" in scatter_df.columns else None,
                hover_data=["handle", "follower_count", "engagement_rate"],
                title="Engagement Rate vs Log₁₀(Followers)",
                labels={
                    "log_followers_plot": "Log₁₀(Followers)",
                    "engagement_rate": "Engagement Rate (%)",
                },
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for scatter plot.")

    with chart_col2:
        st.subheader("📦 Engagement by Category")

        box_df = df_filtered.dropna(subset=["engagement_rate"])
        if not box_df.empty:
            fig = px.box(
                box_df,
                x=niche_col,
                y="engagement_rate",
                color=niche_col,
                title="Engagement Rate Distribution by Niche",
                labels={"engagement_rate": "Engagement Rate (%)"},
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for box plots.")

    st.markdown("---")

    # --- Charts row 2: Hashtags + Automation ---
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.subheader("☁️ Top Hashtags")

        if "top_hashtags" in df_filtered.columns:
            all_tags = []
            for tags_str in df_filtered["top_hashtags"].dropna():
                tags = [t.strip() for t in str(tags_str).split(",") if t.strip()]
                all_tags.extend(tags)

            if all_tags:
                from collections import Counter
                tag_counts = Counter(all_tags).most_common(20)
                tag_df = pd.DataFrame(tag_counts, columns=["Hashtag", "Count"])

                fig = px.bar(
                    tag_df,
                    x="Count",
                    y="Hashtag",
                    orientation="h",
                    title="Top 20 Most Common Hashtags",
                    template="plotly_dark",
                    color="Count",
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(height=450, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hashtag data available.")
        else:
            st.info("No hashtag column in dataset.")

    with chart_col4:
        st.subheader("🤖 Automation Adoption")

        if "automation_flag" in df_filtered.columns:
            auto_counts = df_filtered["automation_flag"].value_counts()
            labels_map = {0: "No Automation", 1: "Uses Automation"}
            auto_df = pd.DataFrame({
                "Status": [labels_map.get(k, str(k)) for k in auto_counts.index],
                "Count": auto_counts.values,
            })

            fig = px.pie(
                auto_df,
                values="Count",
                names="Status",
                title="Automation Tool Usage",
                template="plotly_dark",
                color_discrete_sequence=["#636EFA", "#EF553B"],
                hole=0.4,
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

            # Show common automation evidence
            if "automation_evidence" in df_filtered.columns:
                evidence = df_filtered[df_filtered["automation_evidence"] != ""]["automation_evidence"]
                if not evidence.empty:
                    all_evidence = []
                    for ev in evidence.dropna():
                        all_evidence.extend(str(ev).split("; "))
                    if all_evidence:
                        from collections import Counter
                        ev_counts = Counter(all_evidence).most_common(10)
                        st.markdown("**Most common automation signals:**")
                        for signal, count in ev_counts:
                            st.markdown(f"- `{signal}` ({count})")
        else:
            st.info("No automation data available.")

    st.markdown("---")

    # --- Follower Tier Distribution ---
    if "follower_tier" in df_filtered.columns:
        st.subheader("📊 Follower Tier Distribution")
        tier_col1, tier_col2 = st.columns(2)

        with tier_col1:
            tier_counts = df_filtered["follower_tier"].value_counts()
            fig = px.bar(
                x=tier_counts.index,
                y=tier_counts.values,
                title="Influencers by Follower Tier",
                labels={"x": "Tier", "y": "Count"},
                template="plotly_dark",
                color=tier_counts.index,
                color_discrete_sequence=["#00CC96", "#636EFA", "#EF553B"],
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tier_col2:
            if "engagement_rate" in df_filtered.columns:
                fig = px.box(
                    df_filtered.dropna(subset=["engagement_rate"]),
                    x="follower_tier",
                    y="engagement_rate",
                    color="follower_tier",
                    title="Engagement Rate by Follower Tier",
                    template="plotly_dark",
                    color_discrete_sequence=["#00CC96", "#636EFA", "#EF553B"],
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Download button ---
    st.subheader("⬇️ Download Data")

    csv_buffer = io.StringIO()
    df_filtered.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 Download Filtered Ranked CSV",
        data=csv_buffer.getvalue(),
        file_name="ranked_influencers_filtered.csv",
        mime="text/csv",
    )

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        "*Built for SR NEXT Self-Guided Data Project | "
        "Data collected from public Instagram profiles only*"
    )


if __name__ == "__main__":
    main()
