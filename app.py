import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import to_rgba
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 2025 · Race Analysis",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark theme styling ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;800&family=Barlow:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0d0d0d;
    color: #e8e8e8;
}

/* Main background */
.stApp { background-color: #0d0d0d; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #1e1e1e;
}
[data-testid="stSidebar"] * { color: #cccccc !important; }

/* Header strip */
.f1-header {
    background: linear-gradient(135deg, #e10600 0%, #a00400 50%, #1a0000 100%);
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 4px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.f1-header::before {
    content: "F1";
    position: absolute;
    right: -10px;
    top: -20px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 10rem;
    font-weight: 800;
    color: rgba(255,255,255,0.04);
    line-height: 1;
}
.f1-header h1 {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #ffffff;
    margin: 0;
    line-height: 1;
}
.f1-header p {
    color: rgba(255,255,255,0.65);
    font-size: 0.95rem;
    margin: 0.5rem 0 0;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Section headers */
.section-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #e10600;
    border-left: 3px solid #e10600;
    padding-left: 0.75rem;
    margin: 2rem 0 1rem;
}

/* KPI cards */
.kpi-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1;
    min-width: 130px;
    background: #161616;
    border: 1px solid #222;
    border-top: 3px solid #e10600;
    border-radius: 4px;
    padding: 1rem 1.2rem;
}
.kpi-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin-top: 0.3rem;
}

/* Driver chip */
.driver-chip {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 2px;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    color: #fff;
    margin-right: 4px;
}

/* Dataframe styling */
[data-testid="stDataFrame"] { border: 1px solid #222; border-radius: 4px; }

/* Tabs */
[data-baseweb="tab-list"] { background: #111; border-bottom: 1px solid #222; }
[data-baseweb="tab"] { color: #888 !important; font-family: 'Barlow Condensed', sans-serif; font-size: 1rem; letter-spacing: 0.08em; text-transform: uppercase; }
[data-baseweb="tab"][aria-selected="true"] { color: #e10600 !important; border-bottom-color: #e10600 !important; }

/* Selectbox / filters */
[data-baseweb="select"] { background: #161616 !important; }

/* Metric delta */
[data-testid="stMetric"] { background: #161616; border: 1px solid #222; padding: 1rem; border-radius: 4px; }
[data-testid="stMetricLabel"] { color: #888 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { font-family: 'Barlow Condensed', sans-serif; font-size: 2rem !important; color: #fff !important; }

/* Plotly/matplotlib chart background */
.stPlotlyChart { border: 1px solid #1e1e1e; border-radius: 4px; }

/* Download button */
[data-testid="stDownloadButton"] button {
    background: #e10600 !important;
    color: white !important;
    border: none !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Matplotlib dark theme ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#111111",
    "axes.facecolor":    "#111111",
    "axes.edgecolor":    "#2a2a2a",
    "axes.labelcolor":   "#aaaaaa",
    "axes.titlecolor":   "#dddddd",
    "xtick.color":       "#666666",
    "ytick.color":       "#666666",
    "grid.color":        "#1e1e1e",
    "grid.linewidth":    0.8,
    "text.color":        "#cccccc",
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

F1_RED   = "#e10600"
F1_GOLD  = "#FFD700"
F1_SILVER= "#C0C0C0"
F1_WHITE = "#f0f0f0"

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("f1_2025_results.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    return df

df = load_data()

TEAM_COLORS = df.drop_duplicates("TeamName").set_index("TeamName")["TeamColor"].to_dict()
DRIVER_TEAMS = df.drop_duplicates("FullName").set_index("FullName")["TeamName"].to_dict()

def team_color(team):
    return TEAM_COLORS.get(team, "#888888")

def driver_color(driver):
    return team_color(DRIVER_TEAMS.get(driver, ""))

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏎️ F1 2025 ANALYSIS")
    st.markdown("---")

    all_drivers = sorted(df["FullName"].unique())
    all_teams   = sorted(df["TeamName"].unique())
    all_venues  = df.drop_duplicates("Round").sort_values("Round")["Venue"].tolist()

    selected_drivers = st.multiselect(
        "FILTER DRIVERS",
        all_drivers,
        default=all_drivers[:6],
        help="Select drivers to compare"
    )
    if not selected_drivers:
        selected_drivers = all_drivers

    selected_teams = st.multiselect(
        "FILTER TEAMS",
        all_teams,
        default=all_teams,
    )
    if not selected_teams:
        selected_teams = all_teams

    round_range = st.slider(
        "RACE ROUNDS",
        min_value=1,
        max_value=int(df["Round"].max()),
        value=(1, int(df["Round"].max())),
    )

    st.markdown("---")
    st.markdown("**DATASET STATS**")
    st.markdown(f"📋 **Records:** {len(df):,}")
    st.markdown(f"🏁 **Races:** {df['Venue'].nunique()}")
    st.markdown(f"👤 **Drivers:** {df['FullName'].nunique()}")
    st.markdown(f"🏎️ **Teams:** {df['TeamName'].nunique()}")
    st.markdown("---")
    st.markdown("*Built with FastF1 + Streamlit*")
    st.markdown("*2025 Season Data*")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="f1-header">
    <h1>🏎️ Formula 1 · 2025 Season</h1>
    <p>Performance Analysis &amp; Race Prediction Dashboard · Powered by FastF1</p>
</div>
""", unsafe_allow_html=True)

# ── Filtered data ──────────────────────────────────────────────────────────────
mask = (
    df["FullName"].isin(selected_drivers) &
    df["TeamName"].isin(selected_teams) &
    df["Round"].between(*round_range)
)
dff = df[mask].copy()
df_all_rounds = df[df["Round"].between(*round_range)].copy()  # all drivers, filtered rounds

# ── KPI strip ─────────────────────────────────────────────────────────────────
leader = df.groupby("FullName")["Points"].sum().idxmax()
leader_pts = int(df.groupby("FullName")["Points"].sum().max())
total_races = df["Venue"].nunique()
total_dnfs  = (df["Status"] != "Finished").sum()
most_wins_driver = df.groupby("FullName")["Win"].sum().idxmax()
most_wins_count  = int(df.groupby("FullName")["Win"].sum().max())

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-value">{leader.split()[-1].upper()}</div>
        <div class="kpi-label">Championship Leader</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{leader_pts}</div>
        <div class="kpi-label">Leader Points</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{total_races}</div>
        <div class="kpi-label">Races Completed</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{most_wins_count}</div>
        <div class="kpi-label">Most Wins · {most_wins_driver.split()[-1]}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{total_dnfs}</div>
        <div class="kpi-label">Total DNFs</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{df['FullName'].nunique()}</div>
        <div class="kpi-label">Active Drivers</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 STANDINGS", "🏁 RACE RESULTS", "📈 POINTS PROGRESSION",
    "🔥 HEATMAP & STATS", "🤖 ML PREDICTION", "📥 DATA"
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — STANDINGS
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns(2)

    # Driver standings bar
    with col_a:
        st.markdown('<div class="section-label">Driver Championship</div>', unsafe_allow_html=True)
        driver_pts = (
            df_all_rounds.groupby("FullName")["Points"].sum()
            .sort_values(ascending=False).head(15)
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [driver_color(d) for d in driver_pts.index]
        bars = ax.barh(driver_pts.index[::-1], driver_pts.values[::-1],
                       color=colors[::-1], height=0.65, edgecolor="none")
        # Highlight leader
        bars[-1].set_linewidth(2)
        for bar, val in zip(bars, driver_pts.values[::-1]):
            ax.text(val + 3, bar.get_y() + bar.get_height()/2,
                    str(int(val)), va="center", ha="left",
                    fontsize=9, color="#cccccc", fontweight="600")
        ax.set_xlabel("Points", fontsize=10)
        ax.set_title("Top 15 Drivers — Total Points", fontsize=12, fontweight="700", pad=12)
        ax.set_xlim(0, driver_pts.max() * 1.12)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Constructor standings
    with col_b:
        st.markdown('<div class="section-label">Constructor Championship</div>', unsafe_allow_html=True)
        team_pts = (
            df_all_rounds.groupby("TeamName")["Points"].sum()
            .sort_values(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        tc = [team_color(t) for t in team_pts.index]
        bars = ax.barh(team_pts.index[::-1], team_pts.values[::-1],
                       color=tc[::-1], height=0.6, edgecolor="none")
        for bar, val in zip(bars, team_pts.values[::-1]):
            ax.text(val + 3, bar.get_y() + bar.get_height()/2,
                    str(int(val)), va="center", ha="left",
                    fontsize=9, color="#cccccc", fontweight="600")
        ax.set_xlabel("Points", fontsize=10)
        ax.set_title("Constructor Standings — Total Points", fontsize=12, fontweight="700", pad=12)
        ax.set_xlim(0, team_pts.max() * 1.12)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Wins + Podiums side by side
    st.markdown('<div class="section-label">Wins &amp; Podiums</div>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    with col_c:
        wins = df_all_rounds.groupby("FullName")["Win"].sum().sort_values(ascending=False)
        wins = wins[wins > 0]
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [driver_color(d) for d in wins.index]
        ax.bar(range(len(wins)), wins.values, color=colors, edgecolor="none", width=0.65)
        ax.set_xticks(range(len(wins)))
        ax.set_xticklabels(
            [df[df["FullName"]==n]["Abbreviation"].iloc[0] for n in wins.index],
            fontsize=10, fontweight="700"
        )
        ax.set_ylabel("Wins")
        ax.set_title("Race Wins", fontsize=12, fontweight="700", pad=10)
        ax.grid(axis="y", alpha=0.3)
        # Gold #1 bar
        ax.patches[0].set_edgecolor(F1_GOLD)
        ax.patches[0].set_linewidth(2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_d:
        podiums = df_all_rounds.groupby("FullName")["Podium"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [driver_color(d) for d in podiums.index]
        ax.bar(range(len(podiums)), podiums.values, color=colors, edgecolor="none", width=0.65)
        ax.set_xticks(range(len(podiums)))
        ax.set_xticklabels(
            [df[df["FullName"]==n]["Abbreviation"].iloc[0] for n in podiums.index],
            fontsize=10, fontweight="700"
        )
        ax.set_ylabel("Podiums")
        ax.set_title("Podium Finishes (Top 10 Drivers)", fontsize=12, fontweight="700", pad=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — RACE RESULTS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    venues_in_range = df_all_rounds.drop_duplicates("Round").sort_values("Round")["Venue"].tolist()
    selected_venue = st.selectbox("SELECT RACE", venues_in_range)

    race_df = df[df["Venue"] == selected_venue].copy()
    race_df = race_df.sort_values("Position")

    race_meta = race_df.iloc[0]
    st.markdown(f"""
    <div style="background:#161616;border:1px solid #222;border-left:4px solid #e10600;
    padding:1rem 1.5rem;border-radius:4px;margin-bottom:1rem;">
        <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:800;
        color:#fff;text-transform:uppercase;">{selected_venue}</span>
        <span style="margin-left:1rem;color:#888;font-size:0.9rem;">
        {race_meta['Country']} · Round {int(race_meta['Round'])} · {str(race_meta['Date'])[:10]}</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        finished = race_df[race_df["Status"] == "Finished"].copy()
        dnfs     = race_df[race_df["Status"] != "Finished"].copy()

        # Build display table
        display_rows = []
        for _, row in finished.iterrows():
            pos = int(row["Position"]) if pd.notna(row["Position"]) else "-"
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, f"P{pos}")
            display_rows.append({
                "Pos": medal,
                "Driver": row["Abbreviation"],
                "Full Name": row["FullName"],
                "Team": row["TeamName"],
                "Grid": int(row["GridPosition"]) if pd.notna(row["GridPosition"]) else "-",
                "Laps": int(row["Laps"]),
                "Points": int(row["Points"]),
                "FL": "⚡" if row["FastestLap"] else "",
                "Status": row["Status"],
            })
        for _, row in dnfs.iterrows():
            display_rows.append({
                "Pos": "DNF",
                "Driver": row["Abbreviation"],
                "Full Name": row["FullName"],
                "Team": row["TeamName"],
                "Grid": int(row["GridPosition"]) if pd.notna(row["GridPosition"]) else "-",
                "Laps": int(row["Laps"]),
                "Points": 0,
                "FL": "",
                "Status": row["Status"],
            })

        result_df = pd.DataFrame(display_rows)
        st.dataframe(result_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**RACE STATS**")
        winner_row = finished.iloc[0] if len(finished) else None
        if winner_row is not None:
            st.metric("🏆 Winner", winner_row["Abbreviation"])
            st.metric("Points Awarded", f"{int(finished['Points'].sum())}")
            st.metric("DNFs", f"{len(dnfs)}")
            fl_row = race_df[race_df["FastestLap"] == 1]
            if len(fl_row):
                st.metric("⚡ Fastest Lap", fl_row.iloc[0]["Abbreviation"])
                st.metric("FL Time", fl_row.iloc[0]["FastestLapTime"])

        # Position gain/loss chart
        st.markdown("**Grid → Finish**")
        chart_df = finished.copy().head(10)
        chart_df["Delta"] = chart_df["GridPosition"] - chart_df["Position"]
        fig, ax = plt.subplots(figsize=(4, 4))
        colors_bar = [F1_RED if d > 0 else "#3671C6" if d < 0 else "#555" for d in chart_df["Delta"]]
        ax.barh(chart_df["Abbreviation"][::-1], chart_df["Delta"][::-1],
                color=colors_bar[::-1], height=0.6, edgecolor="none")
        ax.axvline(0, color="#444", linewidth=1)
        ax.set_xlabel("Positions Gained/Lost", fontsize=8)
        ax.set_title("Grid → Finish (Top 10)", fontsize=9, fontweight="700")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — POINTS PROGRESSION
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Cumulative Points Progression</div>', unsafe_allow_html=True)

    # Cumulative points per round
    all_rounds_sorted = sorted(df["Round"].unique())
    progression_data = []
    for driver in selected_drivers:
        cumpts = 0
        for r in all_rounds_sorted:
            row = df[(df["FullName"] == driver) & (df["Round"] == r)]
            cumpts += row["Points"].sum()
            progression_data.append({"Driver": driver, "Round": r, "CumPoints": cumpts})

    prog_df = pd.DataFrame(progression_data)

    # Add venue labels
    venue_map = df.drop_duplicates("Round").set_index("Round")["Venue"].to_dict()

    fig, ax = plt.subplots(figsize=(14, 7))
    for driver in selected_drivers:
        d_data = prog_df[prog_df["Driver"] == driver]
        color = driver_color(driver)
        abbrev = df[df["FullName"] == driver]["Abbreviation"].iloc[0]
        ax.plot(d_data["Round"], d_data["CumPoints"],
                color=color, linewidth=2.2, label=abbrev, zorder=3)
        ax.scatter(d_data["Round"].iloc[-1], d_data["CumPoints"].iloc[-1],
                   color=color, s=60, zorder=5, edgecolors="white", linewidth=0.8)
        ax.annotate(f" {abbrev}",
                    xy=(d_data["Round"].iloc[-1], d_data["CumPoints"].iloc[-1]),
                    fontsize=8.5, color=color, fontweight="700", va="center")

    ax.set_xticks(all_rounds_sorted)
    ax.set_xticklabels(
        [venue_map.get(r, "")[:3].upper() for r in all_rounds_sorted],
        fontsize=8, rotation=45, ha="right"
    )
    ax.set_ylabel("Cumulative Points", fontsize=11)
    ax.set_title("Championship Points Progression — 2025 Season", fontsize=13, fontweight="700", pad=14)
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", alpha=0.15)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8,
              framealpha=0.2, borderpad=0.5, ncol=1)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Points per race (bar grouped by round)
    st.markdown('<div class="section-label">Points Per Race</div>', unsafe_allow_html=True)
    pivot = dff.pivot_table(index="Venue", columns="FullName", values="Points", aggfunc="sum").fillna(0)
    pivot = pivot.reindex(df.drop_duplicates("Round").sort_values("Round")["Venue"])
    pivot = pivot.dropna(how="all")

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(pivot))
    n = len(pivot.columns)
    width = 0.8 / max(n, 1)
    for i, driver in enumerate(pivot.columns):
        abbrev = df[df["FullName"] == driver]["Abbreviation"].iloc[0]
        ax.bar(x + i * width - 0.4 + width/2, pivot[driver],
               width=width * 0.9, color=driver_color(driver),
               edgecolor="none", label=abbrev)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Points")
    ax.set_title("Points per Race (Filtered Drivers)", fontsize=12, fontweight="700", pad=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, framealpha=0.2, ncol=1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — HEATMAP & STATS
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr_cols = ["Position", "Points", "Laps", "Win", "Podium", "PointsFinish", "FastestLap", "GridPosition"]
        corr_data = df_all_rounds[corr_cols].copy()
        corr_data["Position"] = pd.to_numeric(corr_data["Position"], errors="coerce")
        corr = corr_data.corr()

        fig, ax = plt.subplots(figsize=(7, 6))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, annot=True, fmt=".2f", mask=mask,
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            center=0, linewidths=0.5, linecolor="#1a1a1a",
            annot_kws={"size": 9, "weight": "600"},
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        ax.set_title("Feature Correlations", fontsize=12, fontweight="700", pad=12)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown('<div class="section-label">Driver Performance Radar</div>', unsafe_allow_html=True)
        # Radar for top 5 by points
        top5 = df_all_rounds.groupby("FullName")["Points"].sum().nlargest(5).index.tolist()
        metrics = ["Win%", "Podium%", "Points/Race", "Finish%", "FL Count"]
        total_races_each = df_all_rounds.groupby("FullName")["Round"].count()

        radar_data = {}
        for d in top5:
            sub = df_all_rounds[df_all_rounds["FullName"] == d]
            n = len(sub)
            radar_data[d] = [
                sub["Win"].sum() / n * 100,
                sub["Podium"].sum() / n * 100,
                sub["Points"].sum() / n,
                sub[sub["Status"] == "Finished"].shape[0] / n * 100,
                float(sub["FastestLap"].sum()),
            ]

        # Normalize 0–1 per metric
        radar_arr = np.array([radar_data[d] for d in top5])
        radar_norm = (radar_arr - radar_arr.min(0)) / (radar_arr.max(0) - radar_arr.min(0) + 1e-9)

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_facecolor("#111111")
        fig.patch.set_facecolor("#111111")
        ax.spines["polar"].set_color("#2a2a2a")
        ax.tick_params(colors="#555")

        for i, driver in enumerate(top5):
            vals = radar_norm[i].tolist() + radar_norm[i][:1].tolist()
            color = driver_color(driver)
            abbrev = df[df["FullName"] == driver]["Abbreviation"].iloc[0]
            ax.plot(angles, vals, color=color, linewidth=2, label=abbrev)
            ax.fill(angles, vals, color=color, alpha=0.07)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=9, color="#aaa")
        ax.set_yticklabels([])
        ax.set_title("Top 5 Drivers — Performance Radar", fontsize=11, fontweight="700",
                     color="#ddd", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
                  fontsize=9, framealpha=0.2)
        ax.grid(color="#1e1e1e", linewidth=0.8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # DNF analysis
    st.markdown('<div class="section-label">DNF Analysis by Team</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        dnf_df = df_all_rounds[df_all_rounds["Status"] != "Finished"]
        dnf_team = dnf_df.groupby("TeamName").size().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        tc = [team_color(t) for t in dnf_team.index]
        ax.bar(range(len(dnf_team)), dnf_team.values, color=tc, edgecolor="none", width=0.65)
        ax.set_xticks(range(len(dnf_team)))
        ax.set_xticklabels(
            [t.replace(" F1 Team", "").replace(" Racing", "") for t in dnf_team.index],
            fontsize=9, rotation=30, ha="right"
        )
        ax.set_ylabel("DNF Count")
        ax.set_title("DNFs by Constructor", fontsize=12, fontweight="700", pad=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        dnf_reason = dnf_df["Status"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        wedge_colors = [F1_RED, "#3671C6", "#FF8000", "#27F4D2", "#229971",
                        "#FF87BC", "#52E252", "#6692FF", "#B6BABD", "#64C4FF"]
        wedges, texts, autotexts = ax.pie(
            dnf_reason.values,
            labels=dnf_reason.index,
            colors=wedge_colors[:len(dnf_reason)],
            autopct="%1.0f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="#111", linewidth=1.5),
            textprops={"fontsize": 8, "color": "#cccccc"}
        )
        for at in autotexts:
            at.set_color("#fff")
            at.set_fontsize(8)
        ax.set_title("DNF Causes Breakdown", fontsize=12, fontweight="700", pad=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-label">Race Winner Prediction Model</div>', unsafe_allow_html=True)

    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report, accuracy_score
        import sklearn

        # Feature engineering
        ml_df = df.copy()
        ml_df = ml_df[ml_df["Status"] == "Finished"].copy()
        ml_df["Position"] = pd.to_numeric(ml_df["Position"], errors="coerce")
        ml_df = ml_df.dropna(subset=["Position", "GridPosition"])

        # Encode categoricals
        le_driver = LabelEncoder()
        le_team   = LabelEncoder()
        le_venue  = LabelEncoder()

        ml_df["driver_enc"] = le_driver.fit_transform(ml_df["FullName"])
        ml_df["team_enc"]   = le_team.fit_transform(ml_df["TeamName"])
        ml_df["venue_enc"]  = le_venue.fit_transform(ml_df["Venue"])

        # Rolling avg finish position per driver
        ml_df = ml_df.sort_values(["FullName", "Round"])
        ml_df["rolling_pos"] = ml_df.groupby("FullName")["Position"].transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(10)

        features = ["driver_enc", "team_enc", "venue_enc", "GridPosition",
                    "rolling_pos", "Round"]
        target   = "Win"

        X = ml_df[features]
        y = ml_df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Model Configuration**")
            model_choice = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting"])
            n_estimators = st.slider("Estimators", 50, 300, 150, 50)
            max_depth    = st.slider("Max Depth", 2, 10, 5)
            run_model    = st.button("🚀 TRAIN MODEL", use_container_width=True)

        with col2:
            if run_model:
                with st.spinner("Training..."):
                    if model_choice == "Random Forest":
                        model = RandomForestClassifier(
                            n_estimators=n_estimators, max_depth=max_depth,
                            random_state=42, class_weight="balanced"
                        )
                    else:
                        model = GradientBoostingClassifier(
                            n_estimators=n_estimators, max_depth=max_depth, random_state=42
                        )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                st.success(f"✅ Model trained — Test Accuracy: **{acc:.1%}**")

                # Feature importance
                fi = pd.Series(model.feature_importances_, index=features).sort_values()
                fig, ax = plt.subplots(figsize=(7, 4))
                colors_fi = [F1_RED if v == fi.max() else "#3671C6" for v in fi.values]
                ax.barh(fi.index, fi.values, color=colors_fi, edgecolor="none", height=0.6)
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importance", fontsize=12, fontweight="700", pad=10)
                ax.grid(axis="x", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Next race prediction
                st.markdown('<div class="section-label">Next Race Win Probability</div>',
                            unsafe_allow_html=True)

                next_round = int(df["Round"].max()) + 1
                all_drivers_ml = ml_df.drop_duplicates("FullName")[
                    ["FullName", "driver_enc", "team_enc"]
                ]

                next_preds = []
                for _, row in all_drivers_ml.iterrows():
                    venue_enc_val = 0  # placeholder for next venue
                    rolling = ml_df[ml_df["FullName"] == row["FullName"]]["Position"].mean()
                    grid = np.random.randint(1, 21)
                    prob = model.predict_proba(
                        [[row["driver_enc"], row["team_enc"], venue_enc_val,
                          grid, rolling, next_round]]
                    )[0][1]
                    next_preds.append({
                        "Driver": row["FullName"],
                        "Team": df[df["FullName"] == row["FullName"]]["TeamName"].iloc[0],
                        "Win Probability": prob,
                    })

                pred_df = pd.DataFrame(next_preds).sort_values("Win Probability", ascending=False)
                pred_df["Win Probability"] = (pred_df["Win Probability"] * 100).round(1)
                pred_df.insert(0, "Rank", range(1, len(pred_df) + 1))

                fig, ax = plt.subplots(figsize=(10, 5))
                top_pred = pred_df.head(10)
                bar_colors = [driver_color(d) for d in top_pred["Driver"]]
                ax.bar(range(10), top_pred["Win Probability"], color=bar_colors,
                       edgecolor="none", width=0.7)
                ax.set_xticks(range(10))
                ax.set_xticklabels(
                    [df[df["FullName"]==n]["Abbreviation"].iloc[0] for n in top_pred["Driver"]],
                    fontsize=11, fontweight="700"
                )
                ax.set_ylabel("Win Probability (%)")
                ax.set_title("Predicted Win Probability — Next Race", fontsize=13,
                             fontweight="700", pad=12)
                ax.grid(axis="y", alpha=0.3)
                for i, (_, row) in enumerate(top_pred.iterrows()):
                    ax.text(i, row["Win Probability"] + 0.2, f"{row['Win Probability']:.1f}%",
                            ha="center", fontsize=9, color="#ccc", fontweight="600")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.dataframe(pred_df.head(10), use_container_width=True, hide_index=True)
            else:
                st.info("👈 Configure the model on the left and click **TRAIN MODEL** to generate predictions.")

    except ImportError:
        st.warning("Install scikit-learn to enable ML predictions: `pip install scikit-learn`")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — DATA
# ════════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-label">Dataset Preview</div>', unsafe_allow_html=True)

    col_filter, col_search = st.columns([3, 1])
    with col_filter:
        show_cols = st.multiselect(
            "VISIBLE COLUMNS",
            df.columns.tolist(),
            default=["Round", "Venue", "FullName", "Abbreviation", "TeamName",
                     "Position", "GridPosition", "Points", "Laps",
                     "FastestLapTime", "Status", "Win", "Podium"]
        )
    with col_search:
        status_filter = st.selectbox("STATUS", ["All", "Finished", "DNF"])

    display_df = df[df["Round"].between(*round_range)].copy()
    if status_filter == "Finished":
        display_df = display_df[display_df["Status"] == "Finished"]
    elif status_filter == "DNF":
        display_df = display_df[display_df["Status"] != "Finished"]

    if show_cols:
        display_df = display_df[show_cols]

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=450)
    st.caption(f"Showing {len(display_df):,} records")

    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        csv_out = df.to_csv(index=False)
        st.download_button(
            "⬇️ DOWNLOAD FULL CSV",
            data=csv_out,
            file_name="f1_2025_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.5rem;border-top:1px solid #1e1e1e;
text-align:center;color:#444;font-size:0.8rem;letter-spacing:0.08em;text-transform:uppercase;">
    F1 2025 Analysis Dashboard · Built with FastF1 · Pandas · Scikit-Learn · Streamlit
</div>
""", unsafe_allow_html=True)
