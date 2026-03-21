"""
Batch 4 — Enrollment Pattern Deep Dive
Index 16 : Single Gender Dominance Index
Index 17 : Peak Year Index
Index 18 : Recovery Index
Index 19 : Small School Index
Index 20 : Enrollment Concentration Index
Index 21 : Girls Enrollment Ladder
Index 22 : Block Comparison Heatmap Data
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.append('.')
from scripts.config import (
    YEARS, YEAR_PAIRS, BASE_YEAR, CURRENT_YEAR, PREV_YEAR,
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
    if pd.isna(old) or old == 0: return np.nan
    return round((new - old) / old * 100, 2)

def _safe(v):
    return 0 if (pd.isna(v) or v is None) else int(v)

T      = {y: _yr_tag(y) for y in YEARS}
T_BASE = T[BASE_YEAR]
T_CURR = T[CURRENT_YEAR]
T_PREV = T[PREV_YEAR]

ENROLL_COLS = [f"Enroll_{_yr_tag(y)}" for y in YEARS]


# ---------------------------------------------------------------------------
# PIVOT HELPERS
# ---------------------------------------------------------------------------
def build_class_pivot(df_long):
    """Total class enrollment per school per year: C8_2223 ... C12_2526"""
    rows = {}
    for _, r in df_long.iterrows():
        sid = r["school_id"]; tag = _yr_tag(r["year"])
        if sid not in rows: rows[sid] = {"school_id": sid}
        for cls in CLASS_LIST:
            rows[sid][f"C{cls}_{tag}"] = _safe(r.get(f"total_{cls}", 0))
    cp = pd.DataFrame(list(rows.values()))
    for y in YEARS:
        for cls in CLASS_LIST:
            col = f"C{cls}_{_yr_tag(y)}"
            if col not in cp.columns: cp[col] = 0
    return cp


def build_girls_class_pivot(df_long):
    """Girls class enrollment per school per year: G8_2223 ... G12_2526"""
    rows = {}
    for _, r in df_long.iterrows():
        sid = r["school_id"]; tag = _yr_tag(r["year"])
        if sid not in rows: rows[sid] = {"school_id": sid}
        for cls in CLASS_LIST:
            rows[sid][f"G{cls}_{tag}"] = _safe(r.get(f"girls_{cls}", 0))
    gp = pd.DataFrame(list(rows.values()))
    for y in YEARS:
        for cls in CLASS_LIST:
            col = f"G{cls}_{_yr_tag(y)}"
            if col not in gp.columns: gp[col] = 0
    return gp


# ---------------------------------------------------------------------------
# LOAD ALL INPUTS
# ---------------------------------------------------------------------------
def load_inputs(df_long):
    growth_df     = pd.read_csv("outputs/L4_SCHOOL/01_growth_decline.csv")
    stability_df  = pd.read_csv("outputs/L4_SCHOOL/02_stability_index.csv")
    gender_df     = pd.read_csv("outputs/L4_SCHOOL/03_gender_equity.csv")
    transition_df = pd.read_csv("outputs/L4_SCHOOL/04_continuity_index.csv")
    dropout_df    = pd.read_csv("outputs/L4_SCHOOL/05_class_distribution.csv")
    block_health  = pd.read_csv("outputs/L3_BLOCK/06_block_health.csv")
    district_df   = pd.read_csv("outputs/L2_DISTRICT/07_district_performance.csv")
    risk_df       = pd.read_csv("outputs/L4_SCHOOL/08_school_risk_index.csv")
    class_df      = pd.read_csv("outputs/L4_SCHOOL/09_class_strength.csv")
    cluster_df    = pd.read_csv("outputs/L4_SCHOOL/10_school_cluster.csv")

    # Normalise batch2 column names
    for df in [risk_df, class_df, cluster_df]:
        df.rename(columns={"School_ID": "school_id", "School_Name": "school_name",
                            "District": "district", "Block": "block"}, inplace=True, errors="ignore")

    class_pivot = build_class_pivot(df_long)
    girls_pivot = build_girls_class_pivot(df_long)

    # Master: one row per school
    master = growth_df.copy()
    for other in [stability_df, gender_df, transition_df, dropout_df]:
        new_cols = [c for c in other.columns if c not in master.columns]
        master = master.merge(other[["school_id"] + new_cols], on="school_id", how="left")

    master = master.merge(class_pivot, on="school_id", how="left")
    master = master.merge(girls_pivot, on="school_id", how="left")
    master = master.merge(risk_df[["school_id", "Risk_Level", "Risk_Score"]],
                          on="school_id", how="left")
    master = master.merge(
        class_df[["school_id", "Under_Enrolled_Classes"]].drop_duplicates("school_id"),
        on="school_id", how="left")
    master = master.merge(
        cluster_df[["school_id", "Outlier_Type", "Block_Cluster_Type"]].drop_duplicates("school_id"),
        on="school_id", how="left")

    bz = block_health[["District", "Block", "Block_Zone"]].rename(
        columns={"District": "district", "Block": "block"})
    master = master.merge(bz, on=["district", "block"], how="left")

    print(f"[load_inputs] Master ready: {len(master)} schools")
    return master, block_health, district_df, risk_df, df_long


# ---------------------------------------------------------------------------
# INDEX 16: SINGLE GENDER DOMINANCE INDEX
# ---------------------------------------------------------------------------
def calc_gender_dominance(master):
    print("\n[Index 16] Calculating Gender Dominance Index...")

    rows = []
    for _, r in master.iterrows():
        boys = {y: _safe(r.get(f"Boys_{_yr_tag(y)}", 0)) for y in YEARS}
        girls = {y: _safe(r.get(f"Girls_{_yr_tag(y)}", 0)) for y in YEARS}
        totals = {y: boys[y] + girls[y] for y in YEARS}

        gpct = {y: round(girls[y] / (totals[y] + 0.001) * 100, 1) for y in YEARS}

        # Dominance tag (2025-26)
        g_curr = gpct[CURRENT_YEAR]
        if g_curr < 5:       dom_tag = "Boys Only School"
        elif g_curr < 5 and _safe(boys.get(CURRENT_YEAR, 0)) < 5: dom_tag = "Girls Only School"
        elif (100 - g_curr) < 5: dom_tag = "Girls Only School"
        elif g_curr < 25:    dom_tag = "Boys Dominated"
        elif g_curr < 45:    dom_tag = "Slightly Boys Heavy"
        elif g_curr <= 55:   dom_tag = "Balanced"
        elif g_curr <= 75:   dom_tag = "Slightly Girls Heavy"
        else:                dom_tag = "Girls Dominated"

        # Direction change
        g_base = gpct[BASE_YEAR]
        delta = g_curr - g_base
        if delta < -10:   direction = "Moving to Boys Dominant"
        elif delta > 10:  direction = "Moving to Girls Dominant"
        else:             direction = "Stable"

        concern = dom_tag in ("Boys Only School", "Boys Dominated") or \
                  direction == "Moving to Boys Dominant"

        rows.append({
            "District":          r["district"], "Block": r["block"],
            "School_ID":         r["school_id"], "School_Name": r["school_name"],
            f"Total_{T_BASE}":   totals[BASE_YEAR],
            f"Total_{T_CURR}":   totals[CURRENT_YEAR],
            f"Boys_{T_BASE}":    boys[BASE_YEAR],
            f"Girls_{T_BASE}":   girls[BASE_YEAR],
            f"Boys_{T_CURR}":    boys[CURRENT_YEAR],
            f"Girls_{T_CURR}":   girls[CURRENT_YEAR],
            **{f"Girls_pct_{_yr_tag(y)}": gpct[y] for y in YEARS},
            "Gender_Dominance_Tag": dom_tag,
            "Direction_Change":  direction,
            "Concern_Flag":      concern,
        })

    result = pd.DataFrame(rows)
    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    result.to_csv(OUTPUT_PATHS["L4"] + "16_gender_dominance.csv", index=False)

    # Block summary
    block_sum = result.groupby(["District", "Block"]).agg(
        Boys_Only_Schools     = ("Gender_Dominance_Tag", lambda x: (x == "Boys Only School").sum()),
        Girls_Only_Schools    = ("Gender_Dominance_Tag", lambda x: (x == "Girls Only School").sum()),
        Boys_Dominated_Schools= ("Gender_Dominance_Tag", lambda x: (x == "Boys Dominated").sum()),
        Balanced_Schools      = ("Gender_Dominance_Tag", lambda x: (x == "Balanced").sum()),
        Avg_Girls_pct_Base    = (f"Girls_pct_{T_BASE}", "mean"),
        Avg_Girls_pct_Curr    = (f"Girls_pct_{T_CURR}", "mean"),
    ).round(1).reset_index()
    block_sum["Block_Gender_Direction"] = block_sum.apply(
        lambda r: "Improving" if r["Avg_Girls_pct_Curr"] > r["Avg_Girls_pct_Base"] + 2
        else ("Declining" if r["Avg_Girls_pct_Curr"] < r["Avg_Girls_pct_Base"] - 2 else "Stable"), axis=1)
    block_sum.rename(columns={"Avg_Girls_pct_Base": f"Avg_Girls_pct_{T_BASE}",
                               "Avg_Girls_pct_Curr": f"Avg_Girls_pct_{T_CURR}"}, inplace=True)
    os.makedirs(OUTPUT_PATHS["L3"], exist_ok=True)
    block_sum.to_csv(OUTPUT_PATHS["L3"] + "16_block_gender_dominance.csv", index=False)

    boys_only  = int((result["Gender_Dominance_Tag"] == "Boys Only School").sum())
    girls_only = int((result["Gender_Dominance_Tag"] == "Girls Only School").sum())
    moving_boys= int((result["Direction_Change"] == "Moving to Boys Dominant").sum())
    worst_dist = result.groupby("District")[f"Girls_pct_{T_CURR}"].mean().idxmin()

    print(f"  Saved: {OUTPUT_PATHS['L4']}16_gender_dominance.csv")
    print(f"  Boys Only Schools          : {boys_only}")
    print(f"  Girls Only Schools         : {girls_only}")
    print(f"  Moving to Boys Dominant    : {moving_boys}")
    print(f"  District lowest girls %    : {worst_dist}")
    return result


# ---------------------------------------------------------------------------
# INDEX 17: PEAK YEAR INDEX
# ---------------------------------------------------------------------------
def calc_peak_year(master):
    print("\n[Index 17] Calculating Peak Year Index...")

    rows = []
    for _, r in master.iterrows():
        vals = {y: _safe(r.get(f"Enroll_{_yr_tag(y)}", 0)) for y in YEARS}
        valid = {y: v for y, v in vals.items() if v > 0}
        if not valid:
            continue

        peak_yr    = max(valid, key=valid.get)
        peak_val   = valid[peak_yr]
        trough_yr  = min(valid, key=valid.get)
        trough_val = valid[trough_yr]
        curr_val   = vals[CURRENT_YEAR]

        pp_loss = _pct(curr_val - peak_val, peak_val)  # negative means loss
        pp_loss_pct = round(abs(pp_loss), 2) if (not pd.isna(pp_loss) and peak_yr != CURRENT_YEAR) else 0

        # Had a trough between base and current
        mid_vals = [vals[y] for y in YEARS[1:-1]]
        had_trough = any(v < vals[BASE_YEAR] and v < curr_val for v in mid_vals)

        if peak_yr == CURRENT_YEAR:
            tag = "Peak Now"
        elif peak_yr == PREV_YEAR:
            tag = "Recently Peaked"
        elif had_trough and curr_val >= peak_val:
            tag = "V Shape"
        elif not pd.isna(pp_loss) and curr_val > vals.get(PREV_YEAR, 0) and curr_val < peak_val:
            tag = "Recovering"
        elif pp_loss_pct > 30:
            tag = "Post Peak Severe"
        elif pp_loss_pct > 15:
            tag = "Post Peak Moderate"
        elif pp_loss_pct > 5:
            tag = "Post Peak Mild"
        else:
            tag = "Stable"

        rows.append({
            "District":    r["district"], "Block": r["block"],
            "School_ID":   r["school_id"], "School_Name": r["school_name"],
            **{f"Enroll_{_yr_tag(y)}": vals[y] for y in YEARS},
            "Peak_Year":         peak_yr, "Peak_Value":   peak_val,
            "Trough_Year":       trough_yr, "Trough_Value": trough_val,
            "Post_Peak_Loss_pct": pp_loss_pct,
            "Peak_Tag":          tag,
        })

    result = pd.DataFrame(rows)
    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    result.to_csv(OUTPUT_PATHS["L4"] + "17_peak_year.csv", index=False)

    # District summary
    dist_sum = result.groupby("District").agg(
        Peak_Now_Schools       = ("Peak_Tag", lambda x: (x == "Peak Now").sum()),
        Post_Peak_Severe_Schools = ("Peak_Tag", lambda x: (x == "Post Peak Severe").sum()),
        V_Shape_Schools        = ("Peak_Tag", lambda x: (x == "V Shape").sum()),
        Recovering_Schools     = ("Peak_Tag", lambda x: (x == "Recovering").sum()),
        Most_Common_Peak_Year  = ("Peak_Year", lambda x: x.value_counts().idxmax()),
    ).reset_index()
    os.makedirs(OUTPUT_PATHS["L2"], exist_ok=True)
    dist_sum.to_csv(OUTPUT_PATHS["L2"] + "17_district_peak.csv", index=False)

    peak_now   = int((result["Peak_Tag"] == "Peak Now").sum())
    pp_severe  = int((result["Peak_Tag"] == "Post Peak Severe").sum())
    vshape     = int((result["Peak_Tag"] == "V Shape").sum())
    most_peak  = result["Peak_Year"].value_counts().idxmax()

    print(f"  Saved: {OUTPUT_PATHS['L4']}17_peak_year.csv")
    print(f"  Still at Peak (2025-26)    : {peak_now}")
    print(f"  Post Peak Severe           : {pp_severe}")
    print(f"  V Shape Recovery           : {vshape}")
    print(f"  Most common peak year      : {most_peak}")
    return result


# ---------------------------------------------------------------------------
# INDEX 18: RECOVERY INDEX
# ---------------------------------------------------------------------------
def calc_recovery(master):
    print("\n[Index 18] Calculating Recovery Index...")

    rows = []
    for _, r in master.iterrows():
        vals = [_safe(r.get(f"Enroll_{_yr_tag(y)}", 0)) for y in YEARS]
        yoy  = [r.get(f"Growth_{_yr_tag(y1)}_{_yr_tag(y2)}", np.nan)
                for y1, y2 in YEAR_PAIRS]

        had_decline = any(not pd.isna(g) and g < 0 for g in yoy)
        trough_val  = min(v for v in vals if v > 0) if any(v > 0 for v in vals) else 0
        trough_idx  = vals.index(trough_val) if trough_val in vals else 0
        trough_yr   = YEARS[trough_idx]

        curr = vals[-1]; base = vals[0]
        rec_amount = curr - trough_val
        rec_pct    = _pct(curr, trough_val) if trough_val > 0 else np.nan

        # Pattern detection
        g1, g2, g3 = yoy
        all_valid = not any(pd.isna(g) for g in [g1, g2, g3])

        if not had_decline:
            pattern = "No Recovery Needed"
        elif all_valid and g1 < 0 and g2 > 0 and g3 > 0:
            pattern = "Strong Recovery"
        elif all_valid and g1 < 0 and g2 < 0 and g3 > 0:
            pattern = "Late Recovery"
        elif had_decline and curr >= base:
            pattern = "Full Recovery"
        elif had_decline and curr < base and not pd.isna(g3) and g3 > 0:
            pattern = "Partial Recovery"
        elif had_decline and not pd.isna(g3) and g3 < 0:
            pattern = "Still Declining"
        else:
            pattern = "Volatile"

        if pd.isna(rec_pct) or rec_pct <= 0:
            strength = "No Recovery"
        elif rec_pct > 20:    strength = "Strong"
        elif rec_pct > 10:    strength = "Moderate"
        else:                 strength = "Weak"

        rows.append({
            "District":   r["district"], "Block": r["block"],
            "School_ID":  r["school_id"], "School_Name": r["school_name"],
            **{f"Enroll_{_yr_tag(y)}": vals[i] for i, y in enumerate(YEARS)},
            "Had_Decline":       had_decline,
            "Trough_Year":       trough_yr,
            "Trough_Value":      trough_val,
            "Recovery_Amount":   rec_amount,
            "Recovery_pct":      round(rec_pct, 2) if not pd.isna(rec_pct) else np.nan,
            "Recovery_Pattern":  pattern,
            "Recovery_Strength": strength,
        })

    result = pd.DataFrame(rows)
    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    os.makedirs(OUTPUT_PATHS["L5"], exist_ok=True)
    result.to_csv(OUTPUT_PATHS["L4"] + "18_recovery_index.csv", index=False)

    rec_schools = result[result["Recovery_Pattern"].isin(["Full Recovery", "Strong Recovery"])]
    rec_schools.to_csv(OUTPUT_PATHS["L5"] + "recovery_schools.csv", index=False)

    full_rec   = int((result["Recovery_Pattern"] == "Full Recovery").sum())
    strong_rec = int((result["Recovery_Strength"] == "Strong").sum())
    still_dec  = int((result["Recovery_Pattern"] == "Still Declining").sum())
    best_row   = result[result["Recovery_pct"].notna()].nlargest(1, "Recovery_pct")
    best_str   = (f"{best_row.iloc[0]['School_Name'][:40]} ({best_row.iloc[0]['Recovery_pct']:.1f}%)"
                  if not best_row.empty else "N/A")

    print(f"  Saved: {OUTPUT_PATHS['L4']}18_recovery_index.csv")
    print(f"  Full Recovery schools      : {full_rec}")
    print(f"  Strong Recovery schools    : {strong_rec}")
    print(f"  Still Declining            : {still_dec}")
    print(f"  Best recovery story        : {best_str}")
    return result


# ---------------------------------------------------------------------------
# INDEX 19: SMALL SCHOOL INDEX
# ---------------------------------------------------------------------------
def calc_small_school(master):
    print("\n[Index 19] Calculating Small School Index...")

    rows = []
    for _, r in master.iterrows():
        vals = {y: _safe(r.get(f"Enroll_{_yr_tag(y)}", 0)) for y in YEARS}
        curr = vals[CURRENT_YEAR]

        # Size tag
        if curr <= 10:        size_tag = "Ghost School"
        elif curr <= 25:      size_tag = "Micro School"
        elif curr <= 50:      size_tag = "Very Small"
        elif curr <= 200:     size_tag = "Small School"
        elif curr <= 500:     size_tag = "Medium School"
        else:                 size_tag = "Large School"

        # Size trend
        yr_vals = [vals[y] for y in YEARS if vals[y] > 0]
        if len(yr_vals) >= 2:
            if all(yr_vals[i] >= yr_vals[i+1] for i in range(len(yr_vals)-1)):
                size_trend = "Getting Smaller"
            elif all(yr_vals[i] <= yr_vals[i+1] for i in range(len(yr_vals)-1)):
                size_trend = "Growing"
            else:
                size_trend = "Fluctuating"
        else:
            size_trend = "Insufficient Data"

        # Closure risk score
        score = 0
        if curr < 50:  score += 1
        if r.get("Trend_Tag", "") == "Consistent Decline": score += 1
        if r.get("Risk_Level", "") == "HIGH RISK":          score += 1
        if _safe(r.get("Under_Enrolled_Classes", 0)) >= 2:  score += 1

        if score == 4:   cr_tag = "Critical"
        elif score == 3: cr_tag = "High Risk"
        elif score == 2: cr_tag = "At Risk"
        elif score == 1: cr_tag = "Monitor"
        else:            cr_tag = "Safe"

        rows.append({
            "District":   r["district"], "Block": r["block"],
            "School_ID":  r["school_id"], "School_Name": r["school_name"],
            **{f"Total_{_yr_tag(y)}": vals[y] for y in YEARS},
            "Size_Tag":           size_tag,
            "Size_Trend":         size_trend,
            "Closure_Risk_Score": score,
            "Closure_Risk_Tag":   cr_tag,
        })

    result = pd.DataFrame(rows)
    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    os.makedirs(OUTPUT_PATHS["L5"], exist_ok=True)
    result.to_csv(OUTPUT_PATHS["L4"] + "19_small_school.csv", index=False)

    closure_risk = result[result["Closure_Risk_Tag"].isin(["Critical", "High Risk"])]
    closure_risk.to_csv(OUTPUT_PATHS["L5"] + "closure_risk_schools.csv", index=False)

    # Block summary
    block_sum = result.groupby(["District", "Block"]).agg(
        Ghost_Schools         = ("Size_Tag", lambda x: (x == "Ghost School").sum()),
        Micro_Schools         = ("Size_Tag", lambda x: (x == "Micro School").sum()),
        Very_Small_Schools    = ("Size_Tag", lambda x: (x == "Very Small").sum()),
        Critical_Closure_Risk = ("Closure_Risk_Tag", lambda x: (x == "Critical").sum()),
        High_Closure_Risk     = ("Closure_Risk_Tag", lambda x: (x == "High Risk").sum()),
    ).reset_index()
    block_sum.to_csv(OUTPUT_PATHS["L3"] + "19_block_small_schools.csv", index=False)

    ghost   = int((result["Size_Tag"] == "Ghost School").sum())
    micro   = int((result["Size_Tag"] == "Micro School").sum())
    crit    = int((result["Closure_Risk_Tag"] == "Critical").sum())
    most_sm = result[result["Size_Tag"].isin(["Ghost School","Micro School","Very Small"])]\
              .groupby("District").size().idxmax() if len(result) > 0 else "N/A"

    print(f"  Saved: {OUTPUT_PATHS['L4']}19_small_school.csv")
    print(f"  Ghost Schools              : {ghost}")
    print(f"  Micro Schools              : {micro}")
    print(f"  Critical Closure Risk      : {crit}")
    print(f"  District most small schools: {most_sm}")
    return result


# ---------------------------------------------------------------------------
# INDEX 20: ENROLLMENT CONCENTRATION INDEX (HHI)
# ---------------------------------------------------------------------------
def calc_concentration(master):
    print("\n[Index 20] Calculating Enrollment Concentration Index...")

    rows = []
    for _, r in master.iterrows():
        def _hhi(tag):
            total = sum(_safe(r.get(f"C{c}_{tag}", 0)) for c in CLASS_LIST)
            if total == 0: return np.nan
            return round(sum((_safe(r.get(f"C{c}_{tag}", 0)) / total) ** 2 for c in CLASS_LIST), 4)

        hhi_base = _hhi(T_BASE)
        hhi_curr = _hhi(T_CURR)

        total_curr = sum(_safe(r.get(f"C{c}_{T_CURR}", 0)) for c in CLASS_LIST)
        cls_shares = {}
        dom_class  = None; dom_val = -1
        for cls in CLASS_LIST:
            v = _safe(r.get(f"C{cls}_{T_CURR}", 0))
            cls_shares[f"Share_C{cls}"] = round(v / (total_curr + 0.001), 4)
            if v > dom_val: dom_val = v; dom_class = f"Class {cls}"

        top2 = sorted([_safe(r.get(f"C{c}_{T_CURR}", 0)) for c in CLASS_LIST], reverse=True)[:2]
        top2_pct = round(sum(top2) / (total_curr + 0.001) * 100, 1)

        if pd.isna(hhi_curr):    conc_tag = "No Data"
        elif hhi_curr < 0.22:    conc_tag = "Perfectly Balanced"
        elif hhi_curr < 0.30:    conc_tag = "Balanced"
        elif hhi_curr < 0.45:    conc_tag = "Moderately Concentrated"
        elif hhi_curr < 0.65:    conc_tag = "Highly Concentrated"
        else:                    conc_tag = "Extreme Concentration"

        if pd.isna(hhi_base) or pd.isna(hhi_curr):
            hhi_trend = "Unknown"
        elif hhi_curr < hhi_base - 0.05:
            hhi_trend = "Becoming More Balanced"
        elif hhi_curr > hhi_base + 0.05:
            hhi_trend = "Becoming More Concentrated"
        else:
            hhi_trend = "Stable"

        rows.append({
            "District":   r["district"], "Block": r["block"],
            "School_ID":  r["school_id"], "School_Name": r["school_name"],
            **{f"C{c}": _safe(r.get(f"C{c}_{T_CURR}", 0)) for c in CLASS_LIST},
            "Total":          total_curr,
            **cls_shares,
            "HHI_Base":       hhi_base, "HHI_Curr": hhi_curr,
            "HHI_Trend":      hhi_trend,
            "Dominant_Class": dom_class,
            "Top2_Classes_pct": top2_pct,
            "Concentration_Tag": conc_tag,
        })

    result = pd.DataFrame(rows).rename(columns={"HHI_Base": f"HHI_{T_BASE}",
                                                  "HHI_Curr": f"HHI_{T_CURR}"})
    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    result.to_csv(OUTPUT_PATHS["L4"] + "20_concentration_index.csv", index=False)

    dom_top  = result["Dominant_Class"].value_counts().idxmax()
    extreme  = int((result["Concentration_Tag"] == "Extreme Concentration").sum())
    balancing= int((result["HHI_Trend"] == "Becoming More Balanced").sum())

    print(f"  Saved: {OUTPUT_PATHS['L4']}20_concentration_index.csv")
    print(f"  Most common Dominant Class : {dom_top}")
    print(f"  Extreme Concentration      : {extreme} schools")
    print(f"  Becoming more balanced     : {balancing} schools")
    return result


# ---------------------------------------------------------------------------
# INDEX 21: GIRLS ENROLLMENT PROGRESSION
#   Focus: How many girls are enrolled in each class?
#   Is girls enrollment growing as students move up classes?
# ---------------------------------------------------------------------------
def calc_girls_ladder(master, df_long):
    print("\n[Index 21] Calculating Girls Enrollment Progression...")

    # Girls class pivot for all years (for Step3 trend)
    girls_piv = build_girls_class_pivot(df_long)
    m = master.merge(girls_piv, on="school_id", how="left", suffixes=("", "_gp"))

    rows = []
    for _, r in m.iterrows():
        g = {cls: _safe(r.get(f"G{cls}_{T_CURR}", 0)) for cls in CLASS_LIST}
        total_girls = sum(g.values())

        if total_girls == 0:
            rows.append({
                "District": r["district"], "Block": r["block"],
                "School_ID": r["school_id"], "School_Name": r["school_name"],
                **{f"Girls_C{c}": 0 for c in CLASS_LIST},
                "Cont_C8_C9_pct": np.nan, "Cont_C9_C10_pct": np.nan,
                "Cont_C10_C11_pct": np.nan, "Cont_C11_C12_pct": np.nan,
                "Girls_Progression_pct": np.nan, "Weakest_Step": "N/A",
                "C10_C11_Trend": "No Girls Data", "Ladder_Tag": "No Girls Data",
            })
            continue

        s1 = round(g[9]  / (g[8]  + 0.001) * 100, 1)   # Girls C8 -> C9 continuity
        s2 = round(g[10] / (g[9]  + 0.001) * 100, 1)   # Girls C9 -> C10 continuity
        s3 = round(g[11] / (g[10] + 0.001) * 100, 1)   # Girls C10 -> C11 enrollment gap
        s4 = round(g[12] / (g[11] + 0.001) * 100, 1)   # Girls C11 -> C12 continuity
        ov = round(g[12] / (g[8]  + 0.001) * 100, 1)   # Girls overall enrollment progression C8->C12

        steps = {"C8-C9": s1, "C9-C10": s2, "C10-C11": s3, "C11-C12": s4}
        weakest = min(steps, key=steps.get)

        # Class 10→11 enrollment trend across 4 years (biggest enrollment gap point)
        c1011_vals = []
        for y in YEARS:
            tag = _yr_tag(y)
            g10 = _safe(r.get(f"G10_{tag}", 0))
            g11 = _safe(r.get(f"G11_{tag}", 0))
            c1011_vals.append(round(g11 / (g10 + 0.001) * 100, 1))

        valid_c = [v for v in c1011_vals if v < 999]
        if len(valid_c) >= 2:
            if valid_c[-1] > valid_c[0] + 5:   s3_trend = "More Girls Enrolling in C11"
            elif valid_c[-1] < valid_c[0] - 5:  s3_trend = "Fewer Girls Enrolling in C11"
            else:                                s3_trend = "Stable"
        else:
            s3_trend = "Insufficient Data"

        # Enrollment progression tag
        if g[11] == 0 and g[12] == 0:
            l_tag = "Secondary School Only"
        elif ov < 30:
            l_tag = "Collapsed Enrollment"
        elif sum(1 for s in [s1,s2,s3,s4] if s < 70) >= 2:
            l_tag = "Multiple Enrollment Gaps"
        elif s3 < 70:
            l_tag = "Class 10-11 Enrollment Gap"    # Girls drop sharply at Class 11
        elif s1 < 70 or s2 < 70:
            l_tag = "Early Class Enrollment Gap"    # Girls drop in Class 8-10
        elif s4 < 70:
            l_tag = "Class 11-12 Enrollment Gap"
        elif all(s >= 80 for s in [s1, s2, s3, s4]):
            l_tag = "Strong Enrollment"
        else:
            l_tag = "Moderate Enrollment"

        rows.append({
            "District": r["district"], "Block": r["block"],
            "School_ID": r["school_id"], "School_Name": r["school_name"],
            **{f"Girls_C{c}": g[c] for c in CLASS_LIST},
            "Cont_C8_C9_pct":   s1, "Cont_C9_C10_pct":  s2,
            "Cont_C10_C11_pct": s3, "Cont_C11_C12_pct": s4,
            "Girls_Progression_pct": ov,
            "Weakest_Step":          weakest,
            "C10_C11_Trend":         s3_trend,
            "Ladder_Tag":            l_tag,
        })

    result = pd.DataFrame(rows)
    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    os.makedirs(OUTPUT_PATHS["L3"], exist_ok=True)
    os.makedirs(OUTPUT_PATHS["L2"], exist_ok=True)
    result.to_csv(OUTPUT_PATHS["L4"] + "21_girls_ladder.csv", index=False)

    # Block summary
    block_sum = result.groupby(["District", "Block"]).agg(
        Avg_Girls_Progression      = ("Girls_Progression_pct", "mean"),
        Avg_C10_C11_pct            = ("Cont_C10_C11_pct", "mean"),
        Collapsed_Enrollment_Schools = ("Ladder_Tag", lambda x: (x == "Collapsed Enrollment").sum()),
        C1011_Gap_Schools          = ("Ladder_Tag", lambda x: (x == "Class 10-11 Enrollment Gap").sum()),
        Strong_Enrollment_Schools  = ("Ladder_Tag", lambda x: (x == "Strong Enrollment").sum()),
    ).round(1).reset_index()
    block_sum["Block_Girls_Enrollment_Health"] = block_sum.apply(
        lambda r: "Strong"  if r["Avg_Girls_Progression"] > 60 and r["Collapsed_Enrollment_Schools"] == 0
        else ("Weak" if r["Avg_Girls_Progression"] < 25 or r["Collapsed_Enrollment_Schools"] > 2
              else "Moderate"), axis=1)
    block_sum.to_csv(OUTPUT_PATHS["L3"] + "21_block_girls_ladder.csv", index=False)

    # District summary
    dist_sum = result.groupby("District").agg(
        Avg_Girls_Progression  = ("Girls_Progression_pct", "mean"),
        Avg_C10_C11_pct        = ("Cont_C10_C11_pct", "mean"),
        Collapsed_Schools      = ("Ladder_Tag", lambda x: (x == "Collapsed Enrollment").sum()),
    ).round(2).reset_index()
    dist_sum["District_Girls_Enrollment_Rank"] = \
        dist_sum["Avg_Girls_Progression"].rank(ascending=False, method="min").astype(int)
    dist_sum.sort_values("District_Girls_Enrollment_Rank", inplace=True)
    dist_sum.to_csv(OUTPUT_PATHS["L2"] + "21_district_girls_ladder.csv", index=False)

    collapsed  = int((result["Ladder_Tag"] == "Collapsed Enrollment").sum())
    cls11_gap  = int((result["Ladder_Tag"] == "Class 10-11 Enrollment Gap").sum())
    # Exclude secondary-only schools (Girls_C8=0) to avoid inflated progression %
    avg_prog   = round(result[result["Girls_C8"] > 0]["Girls_Progression_pct"].mean(), 1)
    best_d     = dist_sum.iloc[0]["District"]
    worst_d    = dist_sum.iloc[-1]["District"]
    top_weak   = result["Weakest_Step"].value_counts().idxmax()

    print(f"  Saved: {OUTPUT_PATHS['L4']}21_girls_ladder.csv")
    print(f"  Collapsed Enrollment schools    : {collapsed}")
    print(f"  Class 10-11 Enrollment Gap      : {cls11_gap}")
    print(f"  Most common enrollment gap step : {top_weak}")
    print(f"  Avg Girls Enrollment Progression: {avg_prog}%")
    print(f"  Best district (girls)      : {best_d}")
    print(f"  Worst district (girls)     : {worst_d}")
    return result


# ---------------------------------------------------------------------------
# INDEX 22: BLOCK COMPARISON HEATMAP DATA
# ---------------------------------------------------------------------------
def calc_heatmap_data(df_long, master, block_health, risk_df):
    print("\n[Index 22] Calculating Block Heatmap Data...")
    os.makedirs(OUTPUT_PATHS["L3"], exist_ok=True)

    # ---- Heatmap 1: Enrollment by Block + Year ----
    enroll_map = df_long.groupby(["district", "block", "year"]).agg(
        Total_Enrollment = ("total_enrollment", "sum"),
        Avg_Enrollment   = ("total_enrollment", "mean"),
        School_Count     = ("school_id", "nunique"),
    ).round(1).reset_index()
    enroll_map.rename(columns={"district": "District", "block": "Block",
                                "year": "Year"}, inplace=True)

    # YoY growth per block-year
    enroll_pivot = enroll_map.pivot_table(
        index=["District", "Block"], columns="Year",
        values="Total_Enrollment", fill_value=0).reset_index()
    for y1, y2 in YEAR_PAIRS:
        if y1 in enroll_pivot.columns and y2 in enroll_pivot.columns:
            enroll_pivot[f"Growth_{_yr_tag(y1)}_{_yr_tag(y2)}"] = enroll_pivot.apply(
                lambda r: _pct(r[y2], r[y1]), axis=1)

    enroll_map.to_csv(OUTPUT_PATHS["L3"] + "22_heatmap_enrollment.csv", index=False)

    # ---- Heatmap 2: Gender Ratio by Block + Year ----
    df_long["girls_ratio"] = df_long["total_girls"] / (df_long["total_enrollment"] + 0.001)
    gender_map = df_long.groupby(["district", "block", "year"]).agg(
        Avg_Girls_Ratio = ("girls_ratio", "mean"),
    ).round(3).reset_index()
    gender_map.rename(columns={"district": "District", "block": "Block", "year": "Year"}, inplace=True)
    gender_map.to_csv(OUTPUT_PATHS["L3"] + "22_heatmap_gender.csv", index=False)

    # ---- Heatmap 3: Transition proxy (C9/C8 ratio) by Block + Year ----
    df_long["trans_proxy"] = df_long["total_9"] / (df_long["total_8"] + 0.001) * 100
    trans_map = df_long.groupby(["district", "block", "year"]).agg(
        Avg_Trans_8_9 = ("trans_proxy", "mean"),
    ).round(1).reset_index()
    trans_map.rename(columns={"district": "District", "block": "Block", "year": "Year"}, inplace=True)
    trans_map.to_csv(OUTPUT_PATHS["L3"] + "22_heatmap_transition.csv", index=False)

    # ---- Heatmap 4: Risk density by Block ----
    risk_norm = risk_df.copy()
    risk_norm.rename(columns={"School_ID": "school_id", "District": "district",
                               "Block": "block"}, inplace=True, errors="ignore")
    risk_map = risk_norm.groupby(["district", "block"]).agg(
        High_Risk_Count   = ("Risk_Level", lambda x: (x == "HIGH RISK").sum()),
        Medium_Risk_Count = ("Risk_Level", lambda x: (x == "MEDIUM RISK").sum()),
        Low_Risk_Count    = ("Risk_Level", lambda x: (x == "LOW RISK").sum()),
        Total_Schools     = ("school_id", "nunique"),
    ).reset_index()
    risk_map["Risk_Density_pct"] = round(
        risk_map["High_Risk_Count"] / (risk_map["Total_Schools"] + 0.001) * 100, 1)
    bz = block_health[["District", "Block", "Block_Zone"]].rename(
        columns={"District": "district", "Block": "block"})
    risk_map = risk_map.merge(bz, on=["district", "block"], how="left")
    risk_map.rename(columns={"district": "District", "block": "Block"}, inplace=True)
    risk_map.to_csv(OUTPUT_PATHS["L3"] + "22_heatmap_risk.csv", index=False)

    # Print insights
    high_risk_blk = risk_map.nlargest(1, "Risk_Density_pct").iloc[0]
    best_gender   = gender_map[gender_map["Year"] == CURRENT_YEAR]\
                    .nlargest(1, "Avg_Girls_Ratio").iloc[0]
    worst_trans   = trans_map[trans_map["Year"] == CURRENT_YEAR]\
                    .nsmallest(1, "Avg_Trans_8_9").iloc[0]

    print(f"  Saved: 4 heatmap files in {OUTPUT_PATHS['L3']}")
    print(f"  Highest Risk Density Block : {high_risk_blk['Block']} ({high_risk_blk['Risk_Density_pct']:.1f}%)")
    print(f"  Best Girls Ratio Block     : {best_gender['Block']} ({best_gender['Avg_Girls_Ratio']:.3f})")
    print(f"  Worst Transition Block     : {worst_trans['Block']} ({worst_trans['Avg_Trans_8_9']:.1f}%)")

    return enroll_map, gender_map, trans_map, risk_map


# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
def print_final_summary(gd, pk, rv, ss, co, gl, hm_risk):
    boys_only  = int((gd["Gender_Dominance_Tag"] == "Boys Only School").sum())
    girls_only = int((gd["Gender_Dominance_Tag"] == "Girls Only School").sum())
    mov_boys   = int((gd["Direction_Change"] == "Moving to Boys Dominant").sum())

    peak_now   = int((pk["Peak_Tag"] == "Peak Now").sum())
    pp_severe  = int((pk["Peak_Tag"] == "Post Peak Severe").sum())
    vshape     = int((pk["Peak_Tag"] == "V Shape").sum())

    full_rec   = int((rv["Recovery_Pattern"] == "Full Recovery").sum())
    strong_rec = int((rv["Recovery_Strength"] == "Strong").sum())
    still_dec  = int((rv["Recovery_Pattern"] == "Still Declining").sum())

    ghost   = int((ss["Size_Tag"] == "Ghost School").sum())
    crit_cl = int((ss["Closure_Risk_Tag"] == "Critical").sum())

    extreme = int((co["Concentration_Tag"] == "Extreme Concentration").sum())
    dom_cls = co["Dominant_Class"].value_counts().idxmax()

    collapsed = int((gl["Ladder_Tag"] == "Collapsed Enrollment").sum())
    cls11_gap = int((gl["Ladder_Tag"] == "Class 10-11 Enrollment Gap").sum())
    avg_ret   = round(gl[gl["Girls_C8"] > 0]["Girls_Progression_pct"].mean(), 1)

    hr_blk = hm_risk.nlargest(1, "Risk_Density_pct").iloc[0]

    W = 52
    sep  = "+" + "="*W + "+"
    sep2 = "+" + "-"*W + "+"
    def row(t): return "| " + t.ljust(W-2) + " |"

    lines = [
        sep, row("         BATCH 4 ANALYSIS COMPLETE"), sep,
        row("INDEX 16 - GENDER DOMINANCE"),
        row(f"  Boys Only Schools      : {boys_only}"),
        row(f"  Girls Only Schools     : {girls_only}"),
        row(f"  Moving to Boys Dominant: {mov_boys}"),
        sep2,
        row("INDEX 17 - PEAK YEAR"),
        row(f"  Still at Peak (2025-26): {peak_now} schools"),
        row(f"  Post Peak Severe       : {pp_severe} schools"),
        row(f"  V Shape Recovery       : {vshape} schools"),
        sep2,
        row("INDEX 18 - RECOVERY INDEX"),
        row(f"  Full Recovery          : {full_rec} schools"),
        row(f"  Strong Recovery        : {strong_rec} schools"),
        row(f"  Still Declining        : {still_dec} schools"),
        sep2,
        row("INDEX 19 - SMALL SCHOOL"),
        row(f"  Ghost Schools          : {ghost}"),
        row(f"  Critical Closure Risk  : {crit_cl}"),
        sep2,
        row("INDEX 20 - CONCENTRATION"),
        row(f"  Extreme Concentration  : {extreme} schools"),
        row(f"  Most Common Dominant   : {dom_cls}"),
        sep2,
        row("INDEX 21 - GIRLS ENROLLMENT PROGRESSION"),
        row(f"  Collapsed Enrollment   : {collapsed} schools"),
        row(f"  Class 10-11 Enroll Gap : {cls11_gap} schools"),
        row(f"  Avg Girls Progression  : {avg_ret}%"),
        sep2,
        row("INDEX 22 - HEATMAP DATA"),
        row(f"  Highest Risk Block     : {str(hr_blk['Block'])[:30]}"),
        row(f"  Risk Density           : {hr_blk['Risk_Density_pct']:.1f}%"),
        sep,
    ]
    print("\n" + "\n".join(lines))

    saved = [
        f"{OUTPUT_PATHS['L4']}16_gender_dominance.csv",
        f"{OUTPUT_PATHS['L3']}16_block_gender_dominance.csv",
        f"{OUTPUT_PATHS['L4']}17_peak_year.csv",
        f"{OUTPUT_PATHS['L2']}17_district_peak.csv",
        f"{OUTPUT_PATHS['L4']}18_recovery_index.csv",
        f"{OUTPUT_PATHS['L5']}recovery_schools.csv",
        f"{OUTPUT_PATHS['L4']}19_small_school.csv",
        f"{OUTPUT_PATHS['L5']}closure_risk_schools.csv",
        f"{OUTPUT_PATHS['L3']}19_block_small_schools.csv",
        f"{OUTPUT_PATHS['L4']}20_concentration_index.csv",
        f"{OUTPUT_PATHS['L4']}21_girls_ladder.csv",
        f"{OUTPUT_PATHS['L3']}21_block_girls_ladder.csv",
        f"{OUTPUT_PATHS['L2']}21_district_girls_ladder.csv",
        f"{OUTPUT_PATHS['L3']}22_heatmap_enrollment.csv",
        f"{OUTPUT_PATHS['L3']}22_heatmap_gender.csv",
        f"{OUTPUT_PATHS['L3']}22_heatmap_transition.csv",
        f"{OUTPUT_PATHS['L3']}22_heatmap_risk.csv",
    ]
    print("\nFiles saved:")
    for f in saved:
        print(f"  {f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_long = load_data(DATA_FILE)

    master, block_health, district_df, risk_df, _ = load_inputs(df_long)

    gd = calc_gender_dominance(master)
    pk = calc_peak_year(master)
    rv = calc_recovery(master)
    ss = calc_small_school(master)
    co = calc_concentration(master)
    gl = calc_girls_ladder(master, df_long)
    enroll_map, gender_map, trans_map, risk_map = calc_heatmap_data(
        df_long, master, block_health, risk_df)

    print_final_summary(gd, pk, rv, ss, co, gl, risk_map)
