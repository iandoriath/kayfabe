##############################################################################
# TAB 3: Active BK Rankings
##############################################################################
with tab3:
    st.header("Active BK Rankings")
    st.write("Pick a date, and see which wrestlers have competed in the prior 12 months with at least 10 appearances, plus their BK skill at that date.")

    # 1) User picks a date
    default_ranking_date = datetime.date(2025, 1, 1)
    ranking_date = st.date_input("Ranking Date:", default_ranking_date)
    ranking_ts = pd.to_datetime(ranking_date)

    # 2) Define "active" = at least 10 matches within [ranking_ts - 365 days, ranking_ts]
    one_year_ago = ranking_ts - pd.DateOffset(days=365)

    # We look at df_lc to see which wrestlers have 10 or more matches in that time window
    # i.e., count matches per wrestler in the window
    df_active_window = df_lc[(df_lc["date"] >= one_year_ago) & (df_lc["date"] <= ranking_ts)]

    # Group by wrestler_id and count the number of matches
    appearances = df_active_window.groupby("wrestler_id").size()

    # Filter wrestlers with at least 10 matches
    min_matches = 10
    active_wrestler_ids = appearances[appearances >= min_matches].index.unique()

    st.write(f"Found {len(active_wrestler_ids)} wrestlers with **{min_matches} or more** matches in the year prior to {ranking_date}.")

    # 3) For each active wrestler, find the row in df_lc that is "closest" to ranking_ts (or clamp)
    #    We'll do a small function inline:
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
        st.warning("No active wrestlers found with the specified criteria. Try a different date or check your data.")
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
        
        st.write("### Top 100 Active Wrestlers (by BK skill) with 10+ Matches in the Past Year")
        st.dataframe(df_top100_display)
