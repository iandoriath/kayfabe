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
    
    # Construct rows
    row_team1_odds = compute_betting_odds_python(p_team1_win)
    row_team2_odds = compute_betting_odds_python(p_team2_win)
    
    import pandas as pd
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
tab1, tab2 = st.tabs(["BK Skill over Time", "Match Simulator"])

with tab1:
    st.title("Bayesian Kayfabe: A Pro Wrestling Skill Metric")
    
    # 3A) Set up date inputs
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
    
    # 3B) Filter df_merged
    df_vis = df_merged[(df_merged["date"] >= start_ts) & (df_merged["date"] <= end_ts)]
    
    # 3C) Multiselect for wrestlers
    all_names = sorted(df_vis["wrestler_names"].dropna().unique())
    default_list = ["John Cena", "Hulk Hogan", "Rhea Ripley", "The Undertaker"]
    defaults_in_data = sorted(set(all_names).intersection(default_list))
    
    chosen = st.multiselect("Choose wrestlers:", options=all_names, default=defaults_in_data)
    
    # 3D) Plot
    if chosen:
        filtered = df_vis[df_vis["wrestler_names"].isin(chosen)]
        if filtered.empty:
            st.warning("No data for the chosen wrestlers and date range.")
        else:
            fig = px.line(
                filtered,
                x="date",
                y="BK",  # your skill column
                color="wrestler_names",
                title="Kayfabe Wrestler Skill Over Time",
                labels={"date": "Date", "BK": "Bayesian Kayfabe Estimated Skill"}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Pick at least one wrestler from the list above.")
    
    st.markdown("---")
    st.markdown("## Further Reading / Commentary")
    st.markdown("""
A Bayesian Kayfabe Wrestler Skill Metric for a Statistical Understanding of WWE Booking Decisions

Professional wrestling, as portrayed by World Wrestling Entertainment (WWE), 
occupies a unique space in sports and entertainment. ...
    """)

with tab2:
    st.header("Match Simulator")
    st.write("Pick wrestlers for Team 1 and Team 2 (one or more per team), select their match dates, then simulate.")
    
    # 4A) Team selections
    all_names_sim = sorted(wrestler_names["wrestler_names"].dropna().unique())
    
    colA, colB = st.columns(2)
    with colA:
        chosen_team1 = st.multiselect("Team 1 Wrestlers:", options=all_names_sim, default=[])
    with colB:
        chosen_team2 = st.multiselect("Team 2 Wrestlers:", options=all_names_sim, default=[])
    
    # 4B) Each team picks a single date (for simplicity)
    colC, colD = st.columns(2)
    with colC:
        team1_date = st.date_input("Team 1 Date:", datetime.date(2025, 1, 1))
    with colD:
        team2_date = st.date_input("Team 2 Date:", datetime.date(2025, 1, 1))
    
    # 4C) Button => run simulation
    if st.button("Simulate Match"):
        if not chosen_team1 or not chosen_team2:
            st.warning("Please select at least one wrestler in each team.")
        else:
            # Build team lists: (wrestler_name, match_date) in Python terms
            team1_list = [(name, pd.to_datetime(team1_date)) for name in chosen_team1]
            team2_list = [(name, pd.to_datetime(team2_date)) for name in chosen_team2]
            
            try:
                sim_result = simulate_match_python(team1_list, team2_list, df_lc, wrestler_names)
                match_df = sim_result["match_summary"]
                st.write("**Simulation Results**")
                st.dataframe(match_df)
                
                st.write("**mu_diff**:", sim_result["mu_diff"])
                st.write("**sigma_diff**:", sim_result["sigma_diff"])
            
            except Exception as e:
                st.error(f"Error in simulation: {e}")
