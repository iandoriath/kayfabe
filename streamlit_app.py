# streamlit_app.py
import streamlit as st
import pyreadr
import pandas as pd
import plotly.express as px
import datetime
import numpy as np

from math import sqrt
from statistics import NormalDist

########################################
# 1) Load Data from RDS
########################################
st.set_page_config(page_title="Bayesian Kayfabe", layout="wide")

df_lc_rds = pyreadr.read_r("df_lc.Rds")
df_lc = df_lc_rds[None]  # main DataFrame from the R object

wrestler_names_rds = pyreadr.read_r("wrestler_names.Rds")
wrestler_names = wrestler_names_rds[None]

# Convert 'date' to a proper datetime in df_lc (if not already)
df_lc["date"] = pd.to_datetime(df_lc["date"], errors="coerce")
df_lc = df_lc.dropna(subset=["date"])

# We'll also build a merged DataFrame for the chart on Tab 1
df_merged = df_lc.merge(wrestler_names, on="wrestler_id", how="left")
df_merged = df_merged.dropna(subset=["date"])

# Determine global min/max date in the dataset
data_min = df_merged["date"].min()
data_max = df_merged["date"].max()

########################################
# 2) Helper Functions (Python versions)
########################################

def get_wrestler_mu_sigma_python(wrestler_name, match_date, df_lc, wrestler_names):
    """
    Finds the skill distribution row for a given wrestler_name at a given match_date,
    clamping to earliest or latest if needed, or picking whichever date is closest.
    Returns { "wrestler_name", "wrestler_id", "mu", "sigma" }.
    """
    # 1) Find the wrestler_id(s)
    subset_ids = wrestler_names.loc[wrestler_names["wrestler_names"] == wrestler_name, "wrestler_id"].unique()
    if len(subset_ids) == 0:
        raise ValueError(f"No wrestler_id found for name: {wrestler_name}")
    wid = subset_ids[0]  # if multiple, just pick the first

    # 2) Filter df_lc
    df_w = df_lc[df_lc["wrestler_id"] == wid].copy()
    if df_w.empty:
        raise ValueError(f"No skill data in df_lc for wrestler: {wrestler_name}, ID={wid}")
    
    # 3) Sort by date
    df_w = df_w.sort_values("date").reset_index(drop=True)
    
    # 4) Compare match_date to earliest/latest
    earliest_date = df_w["date"].iloc[0]
    latest_date   = df_w["date"].iloc[-1]
    if match_date <= earliest_date:
        chosen_row = df_w.iloc[0]
    elif match_date >= latest_date:
        chosen_row = df_w.iloc[-1]
    else:
        # find row whose date is closest
        time_diffs = (df_w["date"] - match_date).abs()
        idx_min = time_diffs.idxmin()
        chosen_row = df_w.loc[idx_min]
    
    return {
        "wrestler_name": wrestler_name,
        "wrestler_id":   wid,
        "mu":            chosen_row["mu"],
        "sigma":         chosen_row["sigma"],
    }

def get_team_distribution_python(team_list, df_lc, wrestler_names):
    """
    Summation of mu, summation of sigma^2 (for independence).
    team_list = [(wrestler_name, python_date), ...]
    Returns a dict with { "mu", "sigma", "details":[...] }.
    """
    details = []
    sum_mu = 0.0
    sum_var = 0.0
    
    for (wname, wdate) in team_list:
        single = get_wrestler_mu_sigma_python(wname, wdate, df_lc, wrestler_names)
        sum_mu += single["mu"]
        sum_var += single["sigma"]**2
        details.append(single)
    
    return {
        "mu":     sum_mu,
        "sigma":  np.sqrt(sum_var),
        "details": details
    }

def compute_betting_odds_python(prob):
    """
    Convert a probability into decimal, American, and fractional odds string.
    """
    if prob <= 0:
        return {
            "prob": prob,
            "decimal_odds": float("inf"),
            "american_odds": float("inf"),
            "fractional_odds_str": "∞/1"
        }
    elif prob >= 1:
        return {
            "prob": prob,
            "decimal_odds": 1.0,
            "american_odds": float("-inf"),
            "fractional_odds_str": "0/1"
        }
    
    decimal_odds = 1.0 / prob
    if prob >= 0.5:
        american_odds = -100.0 * (prob / (1.0 - prob))
    else:
        american_odds = 100.0 * ((1.0 - prob) / prob)
    
    # approximate fractional as numerator/denominator
    numerator   = round((1 - prob)*100)
    denominator = round(prob*100)
    if denominator == 0:
        denominator = 1
    frac_str = f"{numerator}/{denominator}"
    
    return {
        "prob": prob,
        "decimal_odds": decimal_odds,
        "american_odds": american_odds,
        "fractional_odds_str": frac_str
    }

def simulate_match_python(team1_list, team2_list, df_lc, wrestler_names):
    """
    team1_list: [(name, date), (name, date), ...]
    team2_list: [(name, date), (name, date), ...]
    Returns { "match_summary": pd.DataFrame, "mu_diff":float, "sigma_diff":float }
    """
    dist1 = get_team_distribution_python(team1_list, df_lc, wrestler_names)
    dist2 = get_team_distribution_python(team2_list, df_lc, wrestler_names)
    
    mu_diff = dist1["mu"] - dist2["mu"]
    sigma_diff = sqrt(dist1["sigma"]**2 + dist2["sigma"]**2)
    
    # Probability that Team1 > Team2
    nd = NormalDist(mu_diff, sigma_diff)
    p_team1_win = 1.0 - nd.cdf(0)
    p_team2_win = 1.0 - p_team1_win
    
    import pandas as pd
    row_team1_odds = compute_betting_odds_python(p_team1_win)
    row_team2_odds = compute_betting_odds_python(p_team2_win)
    
    summary_data = [
        {
            "team":              "Team 1",
            "prob":              row_team1_odds["prob"],
            "decimal_odds":      row_team1_odds["decimal_odds"],
            "american_odds":     row_team1_odds["american_odds"],
            "fractional_odds":   row_team1_odds["fractional_odds_str"],
            "team_mu":           dist1["mu"],
            "team_sigma":        dist1["sigma"],
            "composition": " + ".join([
                f'{x["wrestler_name"]}({x["mu"]:.2f})'
                for x in dist1["details"]
            ])
        },
        {
            "team":              "Team 2",
            "prob":              row_team2_odds["prob"],
            "decimal_odds":      row_team2_odds["decimal_odds"],
            "american_odds":     row_team2_odds["american_odds"],
            "fractional_odds":   row_team2_odds["fractional_odds_str"],
            "team_mu":           dist2["mu"],
            "team_sigma":        dist2["sigma"],
            "composition": " + ".join([
                f'{x["wrestler_name"]}({x["mu"]:.2f})'
                for x in dist2["details"]
            ])
        }
    ]
    match_summary = pd.DataFrame(summary_data)
    
    return {
        "match_summary": match_summary,
        "mu_diff":       mu_diff,
        "sigma_diff":    sigma_diff
    }

########################################
# 3) Layout in Tabs
########################################
tab1, tab2, tab3, tab4 = st.tabs(["Historic Wrestler Skill", "Betting Odds", "Active BK Rankings", "The Math"])

##############################################################################
# TAB 1: Historic Wrestler Skill
##############################################################################
with tab1:
    st.title("Bayesian Kayfabe: Wrestler Skill Over Time")
    
    # A) Date range filters
    default_start = datetime.date(1985, 1, 1)
    default_end   = datetime.date.today()
    # clamp to data range
    default_start = max(default_start, data_min.date())
    default_end   = min(default_end, data_max.date())
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date",
            value=default_start,
            min_value=data_min.date(),
            max_value=data_max.date()
        )
    with col2:
        end_date = st.date_input(
            "End date",
            value=default_end,
            min_value=data_min.date(),
            max_value=data_max.date()
        )
    
    start_ts = pd.to_datetime(start_date)
    end_ts   = pd.to_datetime(end_date)
    if start_ts > end_ts:
        st.warning("Start date is after end date. No data to show.")
        st.stop()
    
    # B) Filter df_merged
    df_vis = df_merged[(df_merged["date"] >= start_ts) & (df_merged["date"] <= end_ts)]
    
    # C) Multiselect for wrestlers
    all_names = sorted(df_vis["wrestler_names"].dropna().unique())
    default_list = ["John Cena", "Hulk Hogan", "Rhea Ripley", "The Undertaker"]
    defaults_in_data = sorted(set(all_names).intersection(default_list))
    
    chosen = st.multiselect("Choose wrestlers:", options=all_names, default=defaults_in_data)
    
    # D) Plot
    if chosen:
        filtered = df_vis[df_vis["wrestler_names"].isin(chosen)]
        if filtered.empty:
            st.warning("No data for the chosen wrestlers and date range.")
        else:
            fig = px.line(
                filtered,
                x="date",
                y="mu",  # your skill column
                color="wrestler_names",
                title="Kayfabe Wrestler Skill Over Time",
                labels={"date": "Date", "mu": "Estimated Skill (BK)"}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Pick at least one wrestler from the list above.")



##############################################################################
# TAB 2: Betting Odds & Match Simulator
##############################################################################
# Define all_names_sim for use in TAB 2
all_names_sim = sorted(wrestler_names["wrestler_names"].dropna().unique())

with tab2:
    st.header("Betting Odds & Match Simulator")
    st.write("Pick wrestlers for Team 1 and Team 2, set a unique date for each wrestler, and simulate.")

    # Team 1
    st.write("### Team 1 Selection")
    team1_wrestlers = []
    num_team1 = st.number_input("Number of wrestlers in Team 1:", min_value=1, max_value=10, value=1)
    
    for i in range(num_team1):
        with st.expander(f"Team 1 Wrestler {i + 1}"):
            wrestler = st.selectbox(f"Wrestler {i + 1}:", options=all_names_sim, key=f"team1_wrestler_{i}")
            match_date = st.date_input(f"Match Date for {wrestler}:", datetime.date(2025, 1, 1), key=f"team1_date_{i}")
            team1_wrestlers.append((wrestler, pd.to_datetime(match_date)))

    # Team 2
    st.write("### Team 2 Selection")
    team2_wrestlers = []
    num_team2 = st.number_input("Number of wrestlers in Team 2:", min_value=1, max_value=10, value=1)

    for i in range(num_team2):
        with st.expander(f"Team 2 Wrestler {i + 1}"):
            wrestler = st.selectbox(f"Wrestler {i + 1}:", options=all_names_sim, key=f"team2_wrestler_{i}")
            match_date = st.date_input(f"Match Date for {wrestler}:", datetime.date(2025, 1, 1), key=f"team2_date_{i}")
            team2_wrestlers.append((wrestler, pd.to_datetime(match_date)))

    # Simulate Match
    if st.button("Simulate Match"):
        if not team1_wrestlers or not team2_wrestlers:
            st.warning("Please select at least one wrestler in each team.")
        else:
            try:
                sim_result = simulate_match_python(team1_wrestlers, team2_wrestlers, df_lc, wrestler_names)
                match_df = sim_result["match_summary"]
                st.write("**Simulation Results**")
                st.dataframe(match_df)

                st.write("**mu_diff**:", sim_result["mu_diff"])
                st.write("**sigma_diff**:", sim_result["sigma_diff"])

            except Exception as e:
                st.error(f"Error in simulation: {e}")


##############################################################################
# TAB 3: Active BK Rankings
##############################################################################
with tab3:
    st.header("Active BK Rankings")
    st.write("Pick a date, and see which wrestlers have competed in the prior 12 months "
             "AND have 10+ appearances in that window, plus their BK skill at that date.")

    # 1) User picks a date
    default_ranking_date = datetime.date(2025, 1, 1)
    ranking_date = st.date_input("Ranking Date:", default_ranking_date)
    ranking_ts = pd.to_datetime(ranking_date)

    # 2) Define "active" window = at least one match within [ranking_ts - 365 days, ranking_ts]
    one_year_ago = ranking_ts - pd.DateOffset(days=365)

    # Subset df_lc to that window
    df_active_window = df_lc[
        (df_lc["date"] >= one_year_ago) & (df_lc["date"] <= ranking_ts)
    ]

    # Count the number of appearances (rows) for each wrestler in that window
    df_counts = (
        df_active_window
        .groupby("wrestler_id", as_index=False)
        .size()
        .rename(columns={"size": "n_appearances"})
    )

    # Only keep wrestlers with >=10 appearances
    df_counts_10plus = df_counts[df_counts["n_appearances"] >= 10]

    # active_wrestler_ids is now the subset that meets the 'n >= 10' criterion
    active_wrestler_ids = df_counts_10plus["wrestler_id"].unique()

    st.write(
        f"Found {len(active_wrestler_ids)} wrestlers with 10+ appearances "
        f"in the year prior to {ranking_date}."
    )

    # 3) For each active wrestler, find the row in df_lc that is "closest" to ranking_ts
    def get_mu_sigma_at_date(wid, date_ts):
        sub = df_lc[df_lc["wrestler_id"] == wid].sort_values("date").reset_index(drop=True)
        if sub.empty:
            return None

        earliest = sub["date"].iloc[0]
        latest   = sub["date"].iloc[-1]
        if date_ts <= earliest:
            row = sub.iloc[0]
        elif date_ts >= latest:
            row = sub.iloc[-1]
        else:
            diffs = (sub["date"] - date_ts).abs()
            idxm = diffs.idxmin()
            row = sub.loc[idxm]
        return row

    ranking_rows = []
    for wid in active_wrestler_ids:
        row = get_mu_sigma_at_date(wid, ranking_ts)
        if row is not None:
            ranking_rows.append(row)

    # 4) Build a DataFrame from those rows
    if len(ranking_rows) == 0:
        st.warning("No active wrestlers found. Try a different date or check your data.")
    else:
        df_ranking = pd.DataFrame(ranking_rows)
        # Merge to get the wrestler name
        df_ranking = df_ranking.merge(wrestler_names, on="wrestler_id", how="left")

        # Sort descending by mu
        df_ranking = df_ranking.sort_values("mu", ascending=False)

        # Keep top 100
        df_top100 = df_ranking.head(100)

        # We can display columns like: "wrestler_names", "mu", "sigma", "date"
        # (the date is the row date we ended up using)
        df_top100_display = df_top100[["wrestler_names", "mu", "sigma", "date"]].reset_index(drop=True)

        st.write("### Top 100 Active Wrestlers (≥10 Appearances, by BK skill)")
        st.dataframe(df_top100_display)

# --------------------------------------
# TAB 4: "The Math"
# -------------------------------------
with tab4:
    st.header("Bayesian Kayfabe: A Data-Driven Look at WWE’s Scripted 'Skill'")
    
    st.write("""
    **by Ian DeLorey, MS**

    Professional wrestling occupies a unique gray area: wrestlers are highly athletic, yet each match’s outcome is predetermined. 
    How, then, do we measure someone’s “skill” if a victory is assigned by backstage decisions rather than sporting merit? 
    Enter **Bayesian Kayfabe (BK)**—a statistical framework that treats WWE booking as a signal of a wrestler’s storyline strength.

    ---
    ### Where the Data Comes From
    To create these skill estimates, we assembled historical WWE match results (thanks to the **Cagematch.net** database). 
    Each event has a date, participants, and outcome. Even though outcomes are scripted, they reflect the creative team’s view 
    of who “should” look strong at that moment.

    ---
    ### TrueSkill Through Time (TTT)
    The backbone of our approach is **TrueSkill Through Time (TTT)**—a Bayesian model that updates each competitor’s skill 
    distribution as they compete. Instead of a static rating, **TTT** allows a wrestler’s Bayesian Kayfabe (BK) skill to **drift** over time:

    1. **Multiple wrestlers/teams**: TTT handles tag matches, triple threats, or battle royals by grouping participants into “teams,” 
       then updating each team’s skill based on the match outcome.
    2. **Temporal dimension**: We treat days as discrete steps, allowing a wrestler’s skill to shift between 
       appearances. If WWE starts pushing someone (winning frequently), TTT nudges their skill upward. Lulls or storyline losses 
       pull it down.
    3. **Bayesian inference**: After each match, we get a **posterior** skill estimate, reflecting both prior performances and the 
       new outcome’s evidence.

    Ultimately, TTT yields a series of BK values (the estimated mean skill) for each wrestler on each date. A high **BK** score signals 
    consistently “booked strong” in recent weeks, while a low score indicates storyline losses or midcard status.

    ---
    ### Why It Matters
    - **Captures Momentum**: A wrestler on a hot streak (scripted or otherwise) sees a jump in BK.
    - **Handles Complexities**: Multi-competitor matches, alliances, and random breaks in activity fit naturally into this Bayesian framework.
    - **Quantifies Storyline Strength**: This isn’t real athletic power—it’s a lens into WWE’s creative priorities and pushes at any point in time.

    ---
    ### Conclusion
    By tracking wrestlers’ BK scores, we get a dynamic reading of how WWE itself is “telling the story.” 
    Even though it’s fiction, the data and modeling are very real—and from a statistician’s perspective, 
    TTT is a powerful tool for extracting information from seemingly chaotic booking patterns.

    In short, **Bayesian Kayfabe** uses advanced statistical methods to shine a numerical light on WWE’s scripted outcomes. 
    It’s a fun exploration of sports entertainment analytics, revealing the hidden hand that decides who gets pushed—and who doesn’t—over time.         
    """)

footer = """
<hr>
<div style="text-align: center; font-size: 12px; color: grey;">
    © 2025. All rights reserved.<br>
    Data sourced from <strong>Cagematch.net</strong>.<br>
    <strong>WWE</strong> is a registered trademark of World Wrestling Entertainment, Inc.<br>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
