"""
Batch 2 — Aggregate & Risk Indices
Index 6  : Block Health Index
Index 7  : District Performance Index
Index 8  : School Risk Index (Composite)
Index 9  : Class Strength Index
Index 10 : Cluster / Geography Pattern Index
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.append('.')
from scripts.config import (
    YEARS, YEAR_PAIRS, BASE_YEAR, CURRENT_YEAR,
    CLASS_LIST, OUTPUT_PATHS, RISK_THRESHOLDS, DATA_FILE
)
from scripts.data_loader import load_data

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _yr_tag(y):
    p = y.split("-")
    return p[0][-2:] + p[1][-2:]


def _pct(new, old):
    if old == 0 or pd.isna(old):
        return np.nan
    return round((new - old) / old * 100, 2)


def _ensure_batch1():
    """Run batch1 if CSVs are missing."""
    if not os.path.exists(OUTPUT_PATHS["L4"] + "01_growth_decline.csv"):
        print("batch1 CSVs not found — running batch1_indices.py first...")
        import scripts.batch1_indices  # noqa: triggers __main__ block via import


# ---------------------------------------------------------------------------
# LOAD ALL INPUTS
# ---------------------------------------------------------------------------
def load_inputs():
    _ensure_batch1()
    df_raw = load_data(DATA_FILE)

    growth_df     = pd.read_csv(OUTPUT_PATHS["L4"] + "01_growth_decline.csv")
    stability_df  = pd.read_csv(OUTPUT_PATHS["L4"] + "02_stability_index.csv")
    gender_df     = pd.read_csv(OUTPUT_PATHS["L4"] + "03_gender_equity.csv")
    transition_df = pd.read_csv(OUTPUT_PATHS["L4"] + "04_continuity_index.csv")
    dropout_df    = pd.read_csv(OUTPUT_PATHS["L4"] + "05_class_distribution.csv")

    # Master merged df (one row per school)
    master = growth_df.copy()
    for other in [stability_df, gender_df, transition_df, dropout_df]:
        merge_cols = [c for c in other.columns if c not in master.columns or c in ["school_id"]]
        master = master.merge(other[["school_id"] + [c for c in other.columns
                              if c not in master.columns]],
                              on="school_id", how="left")

    print(f"[load_inputs] Master: {len(master)} schools, {len(master.columns)} columns")
    return df_raw, master, growth_df, stability_df, gender_df, transition_df, dropout_df


# ---------------------------------------------------------------------------
# INDEX 6: BLOCK HEALTH INDEX
# ---------------------------------------------------------------------------
def calc_block_health(master):
    print("\n[Index 6] Calculating Block Health Index...")

    t_curr = _yr_tag(CURRENT_YEAR)
    trans_col = f"Continuity_8_9_{t_curr}"
    has_trans = trans_col in master.columns

    blocks = []
    for (dist, block), grp in master.groupby(["district", "block"]):
        n = len(grp)
        trend_counts = grp["Trend_Tag"].value_counts()

        growth_n   = trend_counts.get("Consistent Growth", 0)
        decline_n  = trend_counts.get("Consistent Decline", 0)
        recovery_n = trend_counts.get("Recovery", 0)
        volatile_n = int((grp["Stability_Label"] == "Volatile").sum()) if "Stability_Label" in grp else 0
        trans_fail = int((grp["Low_Continuity_Flag"] == "Yes").sum()) if "Low_Continuity_Flag" in grp else 0
        high_drop  = int((grp["Enroll_Pattern_Tag"] == "Wide Spread").sum()) if "Enroll_Pattern_Tag" in grp else 0
        girls_decl = int((grp["Girls_Declining_Flag"] == "Yes").sum()) if "Girls_Declining_Flag" in grp else 0

        avg_growth = round(grp["Overall_Growth"].mean(), 2) if "Overall_Growth" in grp else np.nan
        avg_trans  = round(grp[trans_col].mean(), 2) if has_trans and trans_col in grp.columns else np.nan
        avg_girls  = round(grp["Girls_Ratio_Current"].mean(), 3) if "Girls_Ratio_Current" in grp else np.nan

        decline_pct = decline_n / n * 100
        growth_pct  = growth_n  / n * 100

        # Zone classification
        if (decline_pct > 50 or
                (not pd.isna(avg_trans) and avg_trans < RISK_THRESHOLDS["transition_fail"]) or
                (not pd.isna(avg_growth) and avg_growth < -10)):
            zone = "RED"
        elif (30 <= decline_pct <= 50 or
              (not pd.isna(avg_trans) and RISK_THRESHOLDS["transition_fail"] <= avg_trans < 80) or
              (not pd.isna(avg_growth) and -10 <= avg_growth < -5)):
            zone = "ORANGE"
        elif (growth_pct > 60 and
              (pd.isna(avg_trans) or avg_trans > RISK_THRESHOLDS["transition_weak"]) and
              (pd.isna(avg_growth) or avg_growth > 5)):
            zone = "GREEN"
        elif 15 <= decline_pct <= 30 or 15 <= growth_pct <= 30:
            zone = "YELLOW"
        else:
            zone = "YELLOW"

        blocks.append({
            "District":                   dist,
            "Block":                      block,
            "Total_Schools":              n,
            "Growth_Schools":             growth_n,
            "Decline_Schools":            decline_n,
            "Recovery_Schools":           recovery_n,
            "Volatile_Schools":           volatile_n,
            "Low_Continuity_Schools":     trans_fail,
            "Wide_Enroll_Spread_Schools": high_drop,
            "Girls_Declining_Schools":    girls_decl,
            "Avg_Growth_pct":             avg_growth,
            "Avg_Continuity_8_9":          avg_trans,
            "Avg_Girls_Ratio":            avg_girls,
            "Block_Zone":                 zone,
            "Priority_Action_Flag":       zone in ("RED", "ORANGE"),
        })

    block_df = pd.DataFrame(blocks)
    os.makedirs(OUTPUT_PATHS["L3"], exist_ok=True)
    block_df.to_csv(OUTPUT_PATHS["L3"] + "06_block_health.csv", index=False)
    print(f"Saved: {OUTPUT_PATHS['L3']}06_block_health.csv")

    for zone in ["RED", "ORANGE", "GREEN"]:
        names = block_df[block_df["Block_Zone"] == zone]["Block"].tolist()
        print(f"  {zone:6s} blocks ({len(names)}): {', '.join(names) if names else 'None'}")

    return block_df


# ---------------------------------------------------------------------------
# INDEX 7: DISTRICT PERFORMANCE INDEX
# ---------------------------------------------------------------------------
def calc_district_performance(master, block_df):
    print("\n[Index 7] Calculating District Performance Index...")

    t_base = _yr_tag(BASE_YEAR)
    t_curr = _yr_tag(CURRENT_YEAR)

    red_per_dist = block_df[block_df["Block_Zone"] == "RED"].groupby("District").size().rename("Red_Blocks")

    t_curr_trans89  = f"Continuity_8_9_{t_curr}"
    t_curr_trans1011 = f"Continuity_10_11_{t_curr}"
    t_curr_trans1112 = f"Continuity_11_12_{t_curr}"

    dists = []
    for dist, grp in master.groupby("district"):
        n_schools = len(grp)
        n_blocks  = grp["block"].nunique()

        enroll = {}
        for yr in YEARS:
            col = f"Enroll_{_yr_tag(yr)}"
            enroll[yr] = int(grp[col].sum()) if col in grp else 0

        overall_g = _pct(enroll[CURRENT_YEAR], enroll[BASE_YEAR])

        yoy = {}
        for y1, y2 in YEAR_PAIRS:
            yoy[f"YoY_{_yr_tag(y1)}_{_yr_tag(y2)}"] = _pct(enroll[y2], enroll[y1])

        girls_g = round(grp["Girls_Growth_Pct"].mean(), 2) if "Girls_Growth_Pct" in grp else np.nan
        boys_g  = round(grp["Boys_Growth_Pct"].mean(),  2) if "Boys_Growth_Pct"  in grp else np.nan

        avg_t89   = round(grp[t_curr_trans89].mean(),   2) if t_curr_trans89   in grp.columns else np.nan
        avg_t1011 = round(grp[t_curr_trans1011].mean(), 2) if t_curr_trans1011 in grp.columns else np.nan
        avg_t1112 = round(grp[t_curr_trans1112].mean(), 2) if t_curr_trans1112 in grp.columns else np.nan

        red_n = int(red_per_dist.get(dist, 0))
        g_decl = int((grp["Girls_Declining_Flag"] == "Yes").sum()) if "Girls_Declining_Flag" in grp else 0

        # Health tag
        if not pd.isna(overall_g) and overall_g > 10 and red_n == 0:
            tag = "Strong"
        elif not pd.isna(overall_g) and overall_g > 0:
            tag = "Growing"
        elif not pd.isna(overall_g) and overall_g < -10 or red_n > 2:
            tag = "Critical"
        else:
            tag = "Declining"

        row = {
            "District":              dist,
            "Total_Schools":         n_schools,
            "Total_Blocks":          n_blocks,
            f"Enroll_{t_base}":      enroll[BASE_YEAR],
        }
        for yr in YEARS[1:]:
            row[f"Enroll_{_yr_tag(yr)}"] = enroll[yr]
        row.update({
            "Overall_Growth_pct":    overall_g,
            **yoy,
            "Girls_Growth_pct":      girls_g,
            "Boys_Growth_pct":       boys_g,
            "Avg_Continuity_8_9":    avg_t89,
            "Avg_Continuity_10_11":  avg_t1011,
            "Avg_Continuity_11_12":  avg_t1112,
            "Red_Blocks":            red_n,
            "Girls_Declining_Schools": g_decl,
            "District_Health_Tag":   tag,
        })
        dists.append(row)

    dist_df = pd.DataFrame(dists)
    dist_df["District_Rank"] = dist_df["Overall_Growth_pct"].rank(ascending=False, method="min").astype(int)
    dist_df.sort_values("District_Rank", inplace=True)

    os.makedirs(OUTPUT_PATHS["L2"], exist_ok=True)
    dist_df.to_csv(OUTPUT_PATHS["L2"] + "07_district_performance.csv", index=False)
    print(f"Saved: {OUTPUT_PATHS['L2']}07_district_performance.csv")

    print("\n  District Rankings (Growth %):")
    print(f"  {'Rank':<5} {'District':<20} {'Growth%':>8}  Tag")
    print("  " + "-"*45)
    for _, r in dist_df.iterrows():
        g = f"{r['Overall_Growth_pct']:.1f}" if not pd.isna(r['Overall_Growth_pct']) else "N/A"
        print(f"  {int(r['District_Rank']):<5} {r['District']:<20} {g:>8}  {r['District_Health_Tag']}")

    best  = dist_df.iloc[0]
    worst = dist_df.iloc[-1]
    crit  = int((dist_df["District_Health_Tag"] == "Critical").sum())
    print(f"\n  Best    : {best['District']} ({best['Overall_Growth_pct']:.1f}%)")
    print(f"  Worst   : {worst['District']} ({worst['Overall_Growth_pct']:.1f}%)")
    print(f"  Critical: {crit} districts")

    return dist_df


# ---------------------------------------------------------------------------
# INDEX 8: SCHOOL RISK INDEX (COMPOSITE)
# ---------------------------------------------------------------------------
def calc_school_risk(master):
    print("\n[Index 8] Calculating School Risk Index...")

    t_curr = _yr_tag(CURRENT_YEAR)
    t_base = _yr_tag(BASE_YEAR)
    enroll_curr = f"Enroll_{t_curr}"
    enroll_base = f"Enroll_{t_base}"

    rows = []
    for _, r in master.iterrows():
        score = 0
        flags = {}

        # +3 Consistent Decline
        f1 = r.get("Trend_Tag", "") == "Consistent Decline"
        flags["Decline_flag"] = f1
        if f1: score += 3

        # +2 Severe decline
        og = r.get("Overall_Growth", np.nan)
        f2 = not pd.isna(og) and og < -15
        flags["Severe_Decline_flag"] = f2
        if f2: score += 2

        # +2 Low enrollment continuity (few students re-enroll next class)
        f3 = r.get("Low_Continuity_Flag", "No") == "Yes"
        flags["Low_Continuity_flag"] = f3
        if f3: score += 2

        # +2 Wide enrollment spread (Class 8 >> Class 12, enrollment thins sharply)
        f4 = r.get("Enroll_Pattern_Tag", "") == "Wide Spread"
        flags["Wide_Enroll_Spread_flag"] = f4
        if f4: score += 2

        # +2 Low enrollment < 100
        curr_enroll = r.get(enroll_curr, 0)
        f5 = curr_enroll > 0 and curr_enroll < 100
        flags["Low_Enrollment_flag"] = f5
        if f5: score += 2

        # +1 Volatile
        f6 = r.get("Stability_Label", "") == "Volatile"
        flags["Volatile_flag"] = f6
        if f6: score += 1

        # +1 Girls declining
        f7 = r.get("Girls_Declining_Flag", "No") == "Yes"
        flags["Girls_Decline_flag"] = f7
        if f7: score += 1

        # +1 Gender imbalance
        f8 = r.get("Girls_Imbalance_Flag", "No") == "Yes"
        flags["Gender_Imbalance_flag"] = f8
        if f8: score += 1

        # +1 Weak continuity (some students not re-enrolling next class)
        f9 = r.get("Weak_Continuity_Flag", "No") == "Yes"
        flags["Weak_Continuity_flag"] = f9
        if f9: score += 1

        # +1 Very small school < 50
        f10 = curr_enroll > 0 and curr_enroll < 50
        flags["Small_School_flag"] = f10
        if f10: score += 1

        level = "HIGH RISK" if score >= 6 else ("MEDIUM RISK" if score >= 3 else "LOW RISK")

        rows.append({
            "District":          r.get("district", ""),
            "Block":             r.get("block", ""),
            "School_ID":         r.get("school_id", ""),
            "School_Name":       r.get("school_name", ""),
            f"Total_{t_base}":   r.get(enroll_base, 0),
            f"Total_{t_curr}":   curr_enroll,
            "Overall_Growth_pct": og,
            "Risk_Score":        score,
            "Risk_Level":        level,
            **{k: ("Yes" if v else "No") for k, v in flags.items()},
        })

    risk_df = pd.DataFrame(rows).sort_values("Risk_Score", ascending=False)

    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    risk_df.to_csv(OUTPUT_PATHS["L4"] + "08_school_risk_index.csv", index=False)
    print(f"Saved: {OUTPUT_PATHS['L4']}08_school_risk_index.csv")

    # High risk only
    high_risk = risk_df[risk_df["Risk_Level"] == "HIGH RISK"]
    os.makedirs(OUTPUT_PATHS["L5"], exist_ok=True)
    high_risk.to_csv(OUTPUT_PATHS["L5"] + "high_risk_schools.csv", index=False)
    print(f"Saved: {OUTPUT_PATHS['L5']}high_risk_schools.csv")

    hi  = int((risk_df["Risk_Level"] == "HIGH RISK").sum())
    med = int((risk_df["Risk_Level"] == "MEDIUM RISK").sum())
    lo  = int((risk_df["Risk_Level"] == "LOW RISK").sum())
    print(f"\n  HIGH RISK   : {hi} schools")
    print(f"  MEDIUM RISK : {med} schools")
    print(f"  LOW RISK    : {lo} schools")

    print("\n  Top 10 Highest Risk Schools:")
    print(f"  {'School':<35} {'District':<15} Score")
    print("  " + "-"*55)
    for _, r in risk_df.head(10).iterrows():
        nm = str(r["School_Name"])[:33]
        print(f"  {nm:<35} {str(r['District']):<15} {r['Risk_Score']}")

    return risk_df


# ---------------------------------------------------------------------------
# INDEX 9: CLASS STRENGTH INDEX
# ---------------------------------------------------------------------------
def calc_class_strength(df_raw):
    print("\n[Index 9] Calculating Class Strength Index...")

    curr_df = df_raw[df_raw["year"] == CURRENT_YEAR].copy()

    rows = []
    for _, r in curr_df.iterrows():
        c8  = int(r.get("total_8",  0))
        c9  = int(r.get("total_9",  0))
        c10 = int(r.get("total_10", 0))
        c11 = int(r.get("total_11", 0))
        c12 = int(r.get("total_12", 0))
        cls_vals = {"8": c8, "9": c9, "10": c10, "11": c11, "12": c12}
        total    = c8 + c9 + c10 + c11 + c12

        mn   = min(cls_vals.values())
        mx   = max(cls_vals.values())
        imb  = round(mx / (mn + 0.001), 2)
        avg  = round(total / 5, 1) if total > 0 else 0
        dom  = max(cls_vals, key=cls_vals.get)
        under = sum(1 for v in cls_vals.values() if v < 10)
        over  = sum(1 for v in cls_vals.values() if v > 60)

        if under >= 2:
            tag = "Ghost Classes"
        elif over >= 2:
            tag = "Overcrowded"
        elif imb > 4:
            tag = "Severely Imbalanced"
        elif imb > 2:
            tag = "Imbalanced"
        else:
            tag = "Balanced"

        rows.append({
            "District":              r.get("district", ""),
            "Block":                 r.get("block", ""),
            "School_ID":             r.get("school_id", ""),
            "School_Name":           r.get("school_name", ""),
            "C8":  c8,  "C9":  c9,  "C10": c10, "C11": c11, "C12": c12,
            "Total":                 total,
            "Min_Class":             mn,
            "Max_Class":             mx,
            "Avg_Class_Size":        avg,
            "Imbalance_Ratio":       imb,
            "Under_Enrolled_Classes": under,
            "Overcrowded_Classes":   over,
            "Dominant_Class":        f"Class {dom}",
            "Class_Strength_Tag":    tag,
        })

    cls_df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    cls_df.to_csv(OUTPUT_PATHS["L4"] + "09_class_strength.csv", index=False)
    print(f"Saved: {OUTPUT_PATHS['L4']}09_class_strength.csv")

    sev_imb = int((cls_df["Class_Strength_Tag"] == "Severely Imbalanced").sum())
    ghost   = int((cls_df["Class_Strength_Tag"] == "Ghost Classes").sum())
    dom_top = cls_df["Dominant_Class"].value_counts().idxmax()
    print(f"  Severely Imbalanced : {sev_imb} schools")
    print(f"  Ghost Classes       : {ghost} schools")
    print(f"  Most common Dominant Class: {dom_top}")

    return cls_df


# ---------------------------------------------------------------------------
# INDEX 10: CLUSTER / GEOGRAPHY PATTERN INDEX
# ---------------------------------------------------------------------------
def calc_cluster_pattern(master):
    print("\n[Index 10] Calculating Cluster/Geography Pattern Index...")

    block_clusters = []
    school_clusters = []

    for (dist, block), grp in master.groupby(["district", "block"]):
        n = len(grp)
        trend_counts = grp["Trend_Tag"].value_counts()

        growth_n   = trend_counts.get("Consistent Growth", 0)
        decline_n  = trend_counts.get("Consistent Decline", 0)
        recovery_n = trend_counts.get("Recovery", 0)

        # Cluster type
        if decline_n >= 3:
            ctype = "Decline Cluster"
        elif growth_n >= 3:
            ctype = "Growth Pocket"
        elif recovery_n >= 3:
            ctype = "Recovery Cluster"
        elif 1 <= decline_n <= 2 and growth_n >= n // 2:
            ctype = "Isolated Decline"
        elif 1 <= growth_n <= 2 and decline_n >= n // 2:
            ctype = "Isolated Growth"
        else:
            ctype = "Mixed Block"

        block_clusters.append({
            "District":             dist,
            "Block":                block,
            "Cluster_Type":         ctype,
            "Total_Schools_In_Block": n,
            "Declining_Schools":    decline_n,
            "Growing_Schools":      growth_n,
            "Recovery_Schools":     recovery_n,
        })

        for _, r in grp.iterrows():
            stag = r.get("Trend_Tag", "")
            is_growing  = stag == "Consistent Growth"
            is_declining = stag == "Consistent Decline"

            if ctype in ("Decline Cluster",) and is_growing:
                outlier = True; otype = "Bright Spot"
            elif ctype in ("Growth Pocket",) and is_declining:
                outlier = True; otype = "Problem School"
            else:
                outlier = False; otype = "Normal"

            school_clusters.append({
                "District":           r.get("district", ""),
                "Block":              r.get("block", ""),
                "School_ID":          r.get("school_id", ""),
                "School_Name":        r.get("school_name", ""),
                "School_Growth_Tag":  stag,
                "Block_Cluster_Type": ctype,
                "Is_Outlier":         "Yes" if outlier else "No",
                "Outlier_Type":       otype,
            })

    block_clust_df  = pd.DataFrame(block_clusters)
    school_clust_df = pd.DataFrame(school_clusters)

    os.makedirs(OUTPUT_PATHS["L3"], exist_ok=True)
    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    block_clust_df.to_csv(OUTPUT_PATHS["L3"]  + "10_cluster_pattern.csv",  index=False)
    school_clust_df.to_csv(OUTPUT_PATHS["L4"] + "10_school_cluster.csv",   index=False)
    print(f"Saved: {OUTPUT_PATHS['L3']}10_cluster_pattern.csv")
    print(f"Saved: {OUTPUT_PATHS['L4']}10_school_cluster.csv")

    dec_cl = block_clust_df[block_clust_df["Cluster_Type"] == "Decline Cluster"]["Block"].tolist()
    grw_pk = block_clust_df[block_clust_df["Cluster_Type"] == "Growth Pocket"]["Block"].tolist()
    bright = school_clust_df[school_clust_df["Outlier_Type"] == "Bright Spot"]
    prob   = school_clust_df[school_clust_df["Outlier_Type"] == "Problem School"]

    print(f"  Decline Clusters  ({len(dec_cl)}): {', '.join(dec_cl) or 'None'}")
    print(f"  Growth Pockets    ({len(grw_pk)}): {', '.join(grw_pk) or 'None'}")
    print(f"  Bright Spots      ({len(bright)}): {', '.join(bright['School_Name'].tolist()) or 'None'}")
    print(f"  Problem Schools   ({len(prob)}):   {', '.join(prob['School_Name'].tolist()) or 'None'}")

    return block_clust_df, school_clust_df


# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
def print_final_summary(block_df, dist_df, risk_df, cls_df,
                         block_clust_df, school_clust_df):
    red    = int((block_df["Block_Zone"] == "RED").sum())
    orange = int((block_df["Block_Zone"] == "ORANGE").sum())
    yellow = int((block_df["Block_Zone"] == "YELLOW").sum())
    green  = int((block_df["Block_Zone"] == "GREEN").sum())

    best  = dist_df.iloc[0]
    worst = dist_df.iloc[-1]
    crit  = int((dist_df["District_Health_Tag"] == "Critical").sum())

    hi   = int((risk_df["Risk_Level"] == "HIGH RISK").sum())
    med  = int((risk_df["Risk_Level"] == "MEDIUM RISK").sum())
    lo   = int((risk_df["Risk_Level"] == "LOW RISK").sum())

    sev  = int((cls_df["Class_Strength_Tag"] == "Severely Imbalanced").sum())
    ghost= int((cls_df["Class_Strength_Tag"] == "Ghost Classes").sum())

    dc   = int((block_clust_df["Cluster_Type"] == "Decline Cluster").sum())
    gp   = int((block_clust_df["Cluster_Type"] == "Growth Pocket").sum())
    bs   = int((school_clust_df["Outlier_Type"] == "Bright Spot").sum())

    W = 52
    sep = "+" + "="*W + "+"
    def row(txt):
        return "| " + txt.ljust(W-2) + " |"

    bg = f"{best['Overall_Growth_pct']:.1f}" if not pd.isna(best['Overall_Growth_pct']) else "N/A"
    wg = f"{worst['Overall_Growth_pct']:.1f}" if not pd.isna(worst['Overall_Growth_pct']) else "N/A"

    lines = [
        sep,
        row("         BATCH 2 ANALYSIS COMPLETE"),
        sep,
        row("INDEX 6 - BLOCK HEALTH"),
        row(f"  RED    Blocks : {red}"),
        row(f"  ORANGE Blocks : {orange}"),
        row(f"  YELLOW Blocks : {yellow}"),
        row(f"  GREEN  Blocks : {green}"),
        sep,
        row("INDEX 7 - DISTRICT PERFORMANCE"),
        row(f"  Best District  : {best['District']} (+{bg}%)"),
        row(f"  Worst District : {worst['District']} ({wg}%)"),
        row(f"  Critical Districts : {crit}"),
        sep,
        row("INDEX 8 - SCHOOL RISK INDEX"),
        row(f"  HIGH RISK   : {hi} schools"),
        row(f"  MEDIUM RISK : {med} schools"),
        row(f"  LOW RISK    : {lo} schools"),
        sep,
        row("INDEX 9 - CLASS STRENGTH"),
        row(f"  Severely Imbalanced : {sev} schools"),
        row(f"  Ghost Classes       : {ghost} schools"),
        sep,
        row("INDEX 10 - CLUSTER PATTERN"),
        row(f"  Decline Clusters : {dc} blocks"),
        row(f"  Growth Pockets   : {gp} blocks"),
        row(f"  Bright Spots     : {bs} schools"),
        sep,
    ]
    print("\n" + "\n".join(lines))

    saved = [
        OUTPUT_PATHS["L3"] + "06_block_health.csv",
        OUTPUT_PATHS["L2"] + "07_district_performance.csv",
        OUTPUT_PATHS["L4"] + "08_school_risk_index.csv",
        OUTPUT_PATHS["L5"] + "high_risk_schools.csv",
        OUTPUT_PATHS["L4"] + "09_class_strength.csv",
        OUTPUT_PATHS["L3"] + "10_cluster_pattern.csv",
        OUTPUT_PATHS["L4"] + "10_school_cluster.csv",
    ]
    print("\nFiles saved:")
    for f in saved:
        print(f"  {f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_raw, master, growth_df, stability_df, gender_df, transition_df, dropout_df = load_inputs()

    block_df                      = calc_block_health(master)
    dist_df                       = calc_district_performance(master, block_df)
    risk_df                       = calc_school_risk(master)
    cls_df                        = calc_class_strength(df_raw)
    block_clust_df, school_clust_df = calc_cluster_pattern(master)

    print_final_summary(block_df, dist_df, risk_df, cls_df,
                         block_clust_df, school_clust_df)
