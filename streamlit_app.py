import streamlit as st
import pyreadr
import pandas as pd
import plotly.express as px
import datetime

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
            title="Weekly Wrestler Skill Over Time",
            labels={"date": "Date", "mu": "Estimated Skill (mu)"}
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

Ian DeLorey MS

Professional wrestling, as portrayed by World Wrestling Entertainment (WWE), occupies a unique space in sports and entertainment. While the athleticism of the performers is undeniable, the outcomes of matches are ultimately determined by creative decisions rather than pure athletic competition. In industry parlance, the scripted nature of professional wrestling is referred to as “kayfabe,” a term which historically signified the code of secrecy within the wrestling business. Wrestlers are presented to the audience as though they are truly competing, with wins and losses contributing to perceptions of who is “strong” or “skilled.” Yet, because these outcomes are orchestrated, the notion of “skill” in WWE diverges considerably from a purely competitive metric.
Despite its scripted nature, fans, journalists, and analysts have long been interested in quantifying wrestlers’ performance levels or “pushes” as a means of evaluating how WWE’s creative team positions performers. This paper proposes a statistical model—herein called the Bayesian Kayfabe (BK) skill metric—that estimates a wrestler’s “storyline strength” over time, based on match outcomes that appear in the public record. Though these outcomes are fictional in a sporting sense, the company’s booking patterns reflect actual creative and business decisions that shape audience perception, merchandise sales, and other revenue streams. By capturing these decisions quantitatively, we can better analyze how, when, and why certain wrestlers gain or lose prominence in WWE storylines.

""")
