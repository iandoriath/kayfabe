import streamlit as st
import pyreadr
import pandas as pd
import plotly.express as px
import datetime
import numpy as np

def get_wrestler_mu_sigma_python(wrestler_name, match_date, df_lc, wrestler_names):
    """
    For a given wrestler_name + match_date (Python datetime),
    return a dictionary with { "mu": ..., "sigma": ... }.
    """

    # 1) Find the wrestler_id that corresponds to the name
    #    (Assume the 'wrestler_names' DF has columns: wrestler_id, wrestler_names)
    subset_ids = wrestler_names.loc[wrestler_names["wrestler_names"] == wrestler_name, "wrestler_id"].unique()
    if len(subset_ids) == 0:
        raise ValueError(f"No wrestler_id found for name: {wrestler_name}")
    
    # If multiple IDs, just pick the first
    wid = subset_ids[0]
    
    # 2) Filter df_lc for that wid
    df_w = df_lc[df_lc["wrestler_id"] == wid].copy()
    if df_w.empty:
        raise ValueError(f"No skill data in df_lc for wrestler: {wrestler_name}, ID={wid}")
    
    # 3) Sort by date
    df_w = df_w.sort_values("date").reset_index(drop=True)
    
    # 4) Compare match_date to earliest and latest
    earliest_date = df_w["date"].iloc[0]
    latest_date   = df_w["date"].iloc[-1]

    # If match_date < earliest => clamp
    if match_date <= earliest_date:
        chosen_row = df_w.iloc[0]
    elif match_date >= latest_date:
        chosen_row = df_w.iloc[-1]
    else:
        # find row whose date is closest to match_date
        # do an absolute difference
        time_diffs = (df_w["date"] - match_date).abs()
        idx_min = time_diffs.idxmin()
        chosen_row = df_w.loc[idx_min]
    
    return {
        "wrestler_name": wrestler_name,
        "wrestler_id":   wid,
        "mu":            chosen_row["mu"],
        "sigma":         chosen_row["sigma"]
    }





def get_team_distribution_python(team_list, df_lc, wrestler_names):
    """
    team_list: a list of (wrestler_name, date_as_datetime) pairs
        e.g. [("Hulk Hogan", date(1988,9,1)), ("Randy Savage", date(1988,9,1))]
    
    Returns a dict:
      {
        "mu": <float>,
        "sigma": <float>,
        "details": [ list of each wrestler's {wrestler_name, mu, sigma} ]
      }
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
    # Edge cases
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
    
    # Decimal
    decimal_odds = 1.0 / prob
    
    # American
    if prob >= 0.5:
        # negative
        american_odds = -100.0 * (prob / (1.0 - prob))
    else:
        # positive
        american_odds = 100.0 * ((1.0 - prob) / prob)
    
    # Fraction string => approximate integer ratio
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


#### Main Program ####



st.title("Bayesian Kayfabe: Wrestler Skill Over Time")

# 1) Load data from RDS
df_lc_rds = pyreadr.read_r("df_lc.Rds")
df_lc = df_lc_rds[None]  # main DataFrame

wrestler_names_rds = pyreadr.read_r("wrestler_names.Rds")
wrestler_names = wrestler_names_rds[None]

# 2) Merge on wrestler_id so each row has names
df_merged = df_lc.merge(wrestler_names, on="wrestler_id", how="left")

# Confirm that 'date' is recognized as datetime:
df_merged["date"] = pd.to_datetime(df_merged["date"], errors="coerce")
df_merged = df_merged.dropna(subset=["date"])  # remove invalid

# 3) Determine the data’s earliest/latest date
data_min = df_merged["date"].min()
data_max = df_merged["date"].max()

# 4) Define the default start/end
default_start = datetime.date(1985, 1, 1)
default_end   = datetime.date.today()

# Clamp these to the actual data
default_start = max(default_start, data_min.date())
default_end   = min(default_end, data_max.date())

# 5) Two separate date pickers
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

# Convert user-chosen dates to Timestamps
start_ts = pd.to_datetime(start_date)
end_ts   = pd.to_datetime(end_date)

# Optional: if you'd like to ensure start <= end:
if start_ts > end_ts:
    st.warning("Start date is after end date. No data to show.")
    st.stop()

# Filter
df_merged = df_merged[(df_merged["date"] >= start_ts) & (df_merged["date"] <= end_ts)]

# 6) Multiselect for wrestlers
all_names = sorted(df_merged["wrestler_names"].dropna().unique())
default_list = ["John Cena", "Hulk Hogan", "Rhea Ripley", "The Undertaker"]
defaults_in_data = sorted(set(all_names).intersection(default_list))

chosen = st.multiselect(
    "Choose wrestlers:",
    options=all_names,
    default=defaults_in_data
)

# 7) Plot
if chosen:
    filtered = df_merged[df_merged["wrestler_names"].isin(chosen)]
    if filtered.empty:
        st.warning("No data for the chosen wrestlers and date range.")
    else:
        fig = px.line(
            filtered,
            x="date", 
            y="mu",  # your skill column
            color="wrestler_names",
            title="Kayfabe Wrestler Skill Over Time",
            labels={"date": "Date", "BK Skill Metric": "Estimated Skill (BK)"}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Pick at least one wrestler from the list above.")

# 8) Add article text below the app
st.markdown("---")
st.markdown("## Further Reading / Commentary")
st.markdown("""
A Bayesian Kayfabe Wrestler Skill Metric for a Statistical Understanding of WWE Booking Decisions

Professional wrestling, as portrayed by World Wrestling Entertainment (WWE), occupies a unique space in sports and entertainment. While the athleticism of the performers is undeniable, the outcomes of matches are ultimately determined by creative decisions rather than pure athletic competition. In industry parlance, the scripted nature of professional wrestling is referred to as “kayfabe,” a term which historically signified the code of secrecy within the wrestling business. Wrestlers are presented to the audience as though they are truly competing, with wins and losses contributing to perceptions of who is “strong” or “skilled.” Yet, because these outcomes are orchestrated, the notion of “skill” in WWE diverges considerably from a purely competitive metric.

Despite its scripted nature, fans, journalists, and analysts have long been interested in quantifying wrestlers’ performance levels or “pushes” as a means of evaluating how WWE’s creative team positions performers. This paper proposes a statistical model—herein called the Bayesian Kayfabe (BK) skill metric—that estimates a wrestler’s “storyline strength” over time, based on match outcomes that appear in the public record. Though these outcomes are fictional in a sporting sense, the company’s booking patterns reflect actual creative and business decisions that shape audience perception, merchandise sales, and other revenue streams. By capturing these decisions quantitatively, we can better analyze how, when, and why certain wrestlers gain or lose prominence in WWE storylines.

""")


# Then define your tabs:
tab1, tab2 = st.tabs(["Skill Visualization", "Match Simulator"])

with tab1:
    st.title("Bayesian Kayfabe: Wrestler Skill Over Time")
    # (Your existing code for date range, multiselect, plotting, etc.)

with tab2:
    st.header("Match Simulator")
    st.write("Compare two teams of one or more wrestlers, based on their Bayesian Kayfabe skill.")
    
    # 1) Let the user select multiple wrestlers for each team
    all_names = sorted(wrestler_names["wrestler_names"].unique())
    
    colA, colB = st.columns(2)
    with colA:
        chosen_team1 = st.multiselect("Team 1 Wrestlers:", options=all_names, default=[])
    with colB:
        chosen_team2 = st.multiselect("Team 2 Wrestlers:", options=all_names, default=[])
    
    # 2) Let the user pick match dates for each team
    colC, colD = st.columns(2)
    with colC:
        team1_date = st.date_input("Team 1 Date:", datetime.date(2025,1,1))
    with colD:
        team2_date = st.date_input("Team 2 Date:", datetime.date(2025,1,1))
    
    # 3) Only compute if user clicks a button, and both teams have at least 1 wrestler
    if st.button("Simulate Match"):
        if not chosen_team1 or not chosen_team2:
            st.warning("Please select at least one wrestler in each team.")
        else:
            # Build the lists for the simulation: (wrestler_name, date) pairs
            team1_list = [(name, pd.to_datetime(team1_date)) for name in chosen_team1]
            team2_list = [(name, pd.to_datetime(team2_date)) for name in chosen_team2]
            
            try:
                # Call our simulation
                sim_result = simulate_match_python(
                    team1_list=team1_list,
                    team2_list=team2_list,
                    df_lc=df_lc,
                    wrestler_names=wrestler_names
                )
                
                match_df = sim_result["match_summary"]
                st.write("**Simulation Results**")
                st.dataframe(match_df)
                
                st.write("Raw mu_diff:", sim_result["mu_diff"])
                st.write("Raw sigma_diff:", sim_result["sigma_diff"])
                
            except Exception as e:
                st.error(f"Error in simulation: {e}")
