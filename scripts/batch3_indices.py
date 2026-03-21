"""
Batch 3 — Advanced Pattern & Alert Indices
Index 11 : Growth Driver Index
Index 12 : Decline Reason Inference Index
Index 13 : Enrollment Potential Index
Index 14 : Early Warning Index
Index 15 : Segment Split Index
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
    if pd.isna(old) or old == 0:
        return np.nan
    return round((new - old) / old * 100, 2)


def _safe(v):
    return 0 if (pd.isna(v) or v is None) else int(v)


T = {y: _yr_tag(y) for y in YEARS}   # {"2022-23": "2223", ...}
T_BASE = T[BASE_YEAR]
T_CURR = T[CURRENT_YEAR]
T_PREV = T[PREV_YEAR]


# ---------------------------------------------------------------------------
# BUILD CLASS PIVOT: school x (C8_2223 … C12_2526)
# ---------------------------------------------------------------------------
def build_class_pivot(df_long):
    rows = {}
    for _, r in df_long.iterrows():
        sid = r["school_id"]
        tag = _yr_tag(r["year"])
        if sid not in rows:
            rows[sid] = {"school_id": sid}
        for cls in CLASS_LIST:
            rows[sid][f"C{cls}_{tag}"] = _safe(r.get(f"total_{cls}", 0))
    cp = pd.DataFrame(list(rows.values()))
    # Ensure all columns exist
    for yr in YEARS:
        for cls in CLASS_LIST:
            col = f"C{cls}_{_yr_tag(yr)}"
            if col not in cp.columns:
                cp[col] = 0
    return cp


# ---------------------------------------------------------------------------
# LOAD INPUTS
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

    # Normalize School_ID -> school_id in batch2 outputs
    for df in [risk_df, class_df, cluster_df]:
        if "School_ID" in df.columns:
            df.rename(columns={"School_ID": "school_id",
                                "School_Name": "school_name"}, inplace=True)
        if "District" in df.columns:
            df.rename(columns={"District": "district", "Block": "block"}, inplace=True)

    class_pivot = build_class_pivot(df_long)

    # Master: one row per school — merge all batch1 CSVs
    master = growth_df.copy()
    for other in [stability_df, gender_df, transition_df, dropout_df]:
        new_cols = [c for c in other.columns if c not in master.columns]
        master = master.merge(other[["school_id"] + new_cols], on="school_id", how="left")

    master = master.merge(class_pivot,                       on="school_id", how="left")
    master = master.merge(risk_df[["school_id", "Risk_Level", "Risk_Score"]],
                          on="school_id", how="left")
    master = master.merge(cluster_df[["school_id", "Block_Cluster_Type",
                                       "Is_Outlier", "Outlier_Type"]],
                          on="school_id", how="left")

    # Block zone lookup
    bz = block_health[["District", "Block", "Block_Zone"]].copy()
    bz.rename(columns={"District": "district", "Block": "block"}, inplace=True)
    master = master.merge(bz, on=["district", "block"], how="left")

    # Girls ratio in base year
    master["Girls_Ratio_Base"] = master.apply(
        lambda r: round(r["Girls_2223"] / r[f"Enroll_{T_BASE}"], 3)
        if r[f"Enroll_{T_BASE}"] > 0 else np.nan, axis=1
    )

    # Per-school within-year class continuation proxies (latest year)
    for i in range(len(CLASS_LIST) - 1):
        cf, ct = CLASS_LIST[i], CLASS_LIST[i + 1]
        master[f"IntraRatio_{cf}_{ct}"] = master.apply(
            lambda r: round(r[f"C{ct}_{T_CURR}"] / (r[f"C{cf}_{T_CURR}"] + 0.001) * 100, 1), axis=1
        )

    print(f"[load_inputs] Master ready: {len(master)} schools")
    return (master, growth_df, gender_df, transition_df,
            block_health, district_df, risk_df, class_df, cluster_df)


# ---------------------------------------------------------------------------
# INDEX 11: GROWTH DRIVER INDEX
# ---------------------------------------------------------------------------
def calc_growth_driver(master):
    print("\n[Index 11] Calculating Growth Driver Index...")

    high = master[master["Overall_Growth"] > 15].copy()

    rows = []
    for _, r in high.iterrows():
        girls_g = r.get("Girls_Growth_Pct", np.nan)
        boys_g  = r.get("Boys_Growth_Pct",  np.nan)

        # Gender driver
        if pd.isna(girls_g) or pd.isna(boys_g):
            g_driver = "Undetermined"
        elif girls_g > boys_g + 5:
            g_driver = "Girls-Led"
        elif boys_g > girls_g + 5:
            g_driver = "Boys-Led"
        else:
            g_driver = "Balanced"

        # Class driver
        lower_base = sum(_safe(r.get(f"C{c}_{T_BASE}", 0)) for c in [8, 9, 10])
        upper_base = sum(_safe(r.get(f"C{c}_{T_BASE}", 0)) for c in [11, 12])
        lower_curr = sum(_safe(r.get(f"C{c}_{T_CURR}", 0)) for c in [8, 9, 10])
        upper_curr = sum(_safe(r.get(f"C{c}_{T_CURR}", 0)) for c in [11, 12])

        lower_g = _pct(lower_curr, lower_base) or 0
        upper_g = _pct(upper_curr, upper_base) or 0

        if pd.isna(lower_g) and pd.isna(upper_g):
            c_driver = "Undetermined"
        elif not pd.isna(upper_g) and not pd.isna(lower_g) and upper_g > lower_g + 10:
            c_driver = "Upper_Class_Growth"
        elif not pd.isna(lower_g) and not pd.isna(upper_g) and lower_g > upper_g + 10:
            c_driver = "Lower_Class_Growth"
        else:
            c_driver = "All_Round_Growth"

        pattern = f"{g_driver} {c_driver}".replace("_", " ")

        avg_trans = r.get("Avg_Transition_Current", np.nan)
        model = (
            r["Overall_Growth"] > 20 and
            not pd.isna(girls_g) and girls_g > 0 and
            not pd.isna(boys_g)  and boys_g  > 0 and
            (pd.isna(avg_trans) or avg_trans > RISK_THRESHOLDS["transition_weak"]) and
            r.get("Risk_Level", "") == "LOW RISK"
        )

        # Class-wise % change
        cls_changes = {}
        for cls in CLASS_LIST:
            b = _safe(r.get(f"C{cls}_{T_BASE}", 0))
            c = _safe(r.get(f"C{cls}_{T_CURR}", 0))
            cls_changes[f"C{cls}_Change_pct"] = _pct(c, b)

        rows.append({
            "District":          r["district"],
            "Block":             r["block"],
            "School_ID":         r["school_id"],
            "School_Name":       r["school_name"],
            "Overall_Growth_pct": r["Overall_Growth"],
            "Boys_Growth_pct":   boys_g,
            "Girls_Growth_pct":  girls_g,
            **cls_changes,
            "Gender_Driver":     g_driver,
            "Class_Driver":      c_driver,
            "Growth_Pattern_Tag": pattern,
            "Model_School_Flag": model,
        })

    result = pd.DataFrame(rows)

    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    os.makedirs(OUTPUT_PATHS["L5"], exist_ok=True)
    result.to_csv(OUTPUT_PATHS["L4"] + "11_growth_driver.csv", index=False)
    print(f"  Saved: {OUTPUT_PATHS['L4']}11_growth_driver.csv")

    models = result[result["Model_School_Flag"] == True]
    models.sort_values("Overall_Growth_pct", ascending=False)\
          .to_csv(OUTPUT_PATHS["L5"] + "model_schools.csv", index=False)
    print(f"  Saved: {OUTPUT_PATHS['L5']}model_schools.csv")

    # District summary
    dist_sum = result.groupby("District").agg(
        High_Growth_Schools   = ("School_ID", "count"),
        Model_Schools         = ("Model_School_Flag", "sum"),
        Avg_Growth_High_Schools = ("Overall_Growth_pct", "mean"),
    ).reset_index()
    if not result.empty:
        top_pattern = result["Growth_Pattern_Tag"].value_counts().idxmax()
        dist_sum["Most_Common_Growth_Pattern"] = \
            result.groupby("District")["Growth_Pattern_Tag"]\
                  .agg(lambda x: x.value_counts().idxmax()).values
    dist_sum.to_csv(OUTPUT_PATHS["L2"] + "11_district_growth_drivers.csv", index=False)
    print(f"  Saved: {OUTPUT_PATHS['L2']}11_district_growth_drivers.csv")

    top_pattern = result["Growth_Pattern_Tag"].value_counts().idxmax() if not result.empty else "N/A"
    print(f"  High Growth schools : {len(result)}")
    print(f"  Model Schools       : {int(models.shape[0])}")
    print(f"  Top Growth Pattern  : {top_pattern}")
    if not models.empty:
        print("  Top 5 Model Schools:")
        for _, r in models.head(5).iterrows():
            print(f"    {r['School_Name'][:40]:<42} {r['Overall_Growth_pct']:>6.1f}%  {r['District']}")

    return result


# ---------------------------------------------------------------------------
# INDEX 12: DECLINE REASON INFERENCE INDEX
# ---------------------------------------------------------------------------
def calc_decline_reason(master):
    print("\n[Index 12] Calculating Decline Reason Inference Index...")

    declining = master[master["Overall_Growth"] < -5].copy()

    rows = []
    for _, r in declining.iterrows():
        reasons, confidences = [], []

        girls_g = r.get("Girls_Growth_Pct", np.nan)
        boys_g  = r.get("Boys_Growth_Pct",  np.nan)

        # Per-year growth for shock detection
        yoy_vals = [r.get(f"Growth_{_yr_tag(y1)}_{_yr_tag(y2)}", np.nan)
                    for y1, y2 in YEAR_PAIRS]

        # Classes declined?
        cls_declined = sum(1 for c in CLASS_LIST
                           if _safe(r.get(f"C{c}_{T_CURR}", 0)) <
                              _safe(r.get(f"C{c}_{T_BASE}", 0)))

        lower_base = sum(_safe(r.get(f"C{c}_{T_BASE}", 0)) for c in [8, 9, 10])
        upper_base = sum(_safe(r.get(f"C{c}_{T_BASE}", 0)) for c in [11, 12])
        lower_curr = sum(_safe(r.get(f"C{c}_{T_CURR}", 0)) for c in [8, 9, 10])
        upper_curr = sum(_safe(r.get(f"C{c}_{T_CURR}", 0)) for c in [11, 12])

        # Rule 1 — MIGRATION / POPULATION SHIFT
        both_g_decl = (not pd.isna(girls_g) and girls_g < 0 and
                       not pd.isna(boys_g)  and boys_g  < 0)
        if both_g_decl and cls_declined >= 3:
            conf = "HIGH" if cls_declined == 5 else "MEDIUM"
            reasons.append("Migration/Population Shift")
            confidences.append(conf)

        # Rule 2 — SOCIAL / GENDER BARRIER
        if (not pd.isna(girls_g) and not pd.isna(boys_g) and
                girls_g < boys_g - 10):
            conf = "HIGH" if girls_g < -20 else "MEDIUM"
            reasons.append("Social/Gender Barrier")
            confidences.append(conf)

        # Rule 3 — HIGHER EDUCATION TRANSITION ISSUE
        lower_stable = _pct(lower_curr, lower_base)
        upper_g      = _pct(upper_curr, upper_base)
        trans_1011   = r.get(f"IntraRatio_10_11", np.nan)
        if (not pd.isna(upper_g) and upper_g < -5 and
                not pd.isna(lower_stable) and lower_stable > -5):
            reasons.append("Higher Education Transition Issue")
            confidences.append("HIGH")
        elif not pd.isna(trans_1011) and trans_1011 < RISK_THRESHOLDS["transition_fail"]:
            reasons.append("Higher Education Transition Issue")
            confidences.append("HIGH")

        # Rule 4 — LOW ENROLLMENT AT BOARD EXAM CLASSES (Class 10/11 boundary)
        r9_10  = r.get("IntraRatio_9_10",  np.nan)
        r10_11 = r.get("IntraRatio_10_11", np.nan)
        if (not pd.isna(r9_10)  and r9_10  < 70) or \
           (not pd.isna(r10_11) and r10_11 < 70):
            drop_val = min(v for v in [r9_10, r10_11] if not pd.isna(v))
            conf = "HIGH" if drop_val < 60 else "MEDIUM"
            reasons.append("Low Enrollment at Board Class Boundary")
            confidences.append(conf)

        # Rule 5 — ECONOMIC / CHILD LABOUR
        if (not pd.isna(boys_g) and not pd.isna(girls_g) and
                boys_g < girls_g - 10):
            reasons.append("Economic/Child Labour")
            confidences.append("MEDIUM")

        # Rule 6 — INFRASTRUCTURE / EXTERNAL SHOCK
        valid_yoy = [v for v in yoy_vals if not pd.isna(v)]
        if any(v < -25 for v in valid_yoy):
            min_drop = min(valid_yoy)
            all_gradual = all(v > -10 for v in valid_yoy if v != min_drop)
            if all_gradual:
                reasons.append("Infrastructure/External Shock")
                confidences.append("MEDIUM")

        # Rule 7 — UPPER CLASS ONLY DECLINE
        lower_ok = not pd.isna(lower_stable) and lower_stable > -5
        if lower_ok and not pd.isna(upper_g) and upper_g < -10:
            reasons.append("Upper Class Only Decline")
            confidences.append("MEDIUM")

        if not reasons:
            reasons.append("Undetermined")
            confidences.append("LOW")

        # Primary reason = first HIGH confidence, else first overall
        high_idx = next((i for i, c in enumerate(confidences) if c == "HIGH"), 0)
        primary  = reasons[high_idx]
        conf_lvl = confidences[high_idx]

        cls_affected = ", ".join(
            f"C{c}" for c in CLASS_LIST
            if _safe(r.get(f"C{c}_{T_CURR}", 0)) < _safe(r.get(f"C{c}_{T_BASE}", 0))
        ) or "None"
        gen_affected = []
        if not pd.isna(girls_g) and girls_g < 0: gen_affected.append("Girls")
        if not pd.isna(boys_g)  and boys_g  < 0: gen_affected.append("Boys")

        rows.append({
            "District":           r["district"],
            "Block":              r["block"],
            "School_ID":          r["school_id"],
            "School_Name":        r["school_name"],
            "Overall_Growth_pct": r["Overall_Growth"],
            "Girls_Growth_pct":   girls_g,
            "Boys_Growth_pct":    boys_g,
            "All_Reasons_Detected": ", ".join(reasons),
            "Primary_Reason":     primary,
            "Confidence_Level":   conf_lvl,
            "Classes_Affected":   cls_affected,
            "Genders_Affected":   ", ".join(gen_affected) or "None",
        })

    result = pd.DataFrame(rows)

    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    result.to_csv(OUTPUT_PATHS["L4"] + "12_decline_reason.csv", index=False)
    print(f"  Saved: {OUTPUT_PATHS['L4']}12_decline_reason.csv")

    # Block summary
    block_sum = result.groupby(["District", "Block"]).agg(
        Total_Declining_Schools = ("School_ID", "count"),
        Most_Common_Reason      = ("Primary_Reason",
                                   lambda x: x.value_counts().idxmax()),
    ).reset_index()
    reason_dist = result.groupby(["District", "Block", "Primary_Reason"])\
                        .size().reset_index(name="Count")
    block_sum.to_csv(OUTPUT_PATHS["L3"] + "12_block_decline_reasons.csv", index=False)
    print(f"  Saved: {OUTPUT_PATHS['L3']}12_block_decline_reasons.csv")

    print(f"\n  Declining schools analyzed : {len(result)}")
    print("  Primary Reason Distribution:")
    for reason, cnt in result["Primary_Reason"].value_counts().items():
        print(f"    {reason:<35}: {cnt}")

    mig_blocks = block_sum[block_sum["Most_Common_Reason"] ==
                           "Migration/Population Shift"]["Block"].tolist()
    gen_blocks = block_sum[block_sum["Most_Common_Reason"] ==
                           "Social/Gender Barrier"]["Block"].tolist()
    print(f"\n  Blocks: Migration primary   ({len(mig_blocks)}): {', '.join(mig_blocks[:5])}" +
          (" ..." if len(mig_blocks) > 5 else ""))
    print(f"  Blocks: Gender Barrier primary ({len(gen_blocks)}): {', '.join(gen_blocks[:5])}" +
          (" ..." if len(gen_blocks) > 5 else ""))

    return result


# ---------------------------------------------------------------------------
# INDEX 13: ENROLLMENT POTENTIAL INDEX
# ---------------------------------------------------------------------------
def calc_enrollment_potential(master):
    print("\n[Index 13] Calculating Enrollment Potential Index...")

    rows = []
    for _, r in master.iterrows():
        score = 0

        # +1 Rising in latest year
        yoy_latest = r.get(f"Growth_{_yr_tag(PREV_YEAR)}_{T_CURR}", np.nan)
        c1 = not pd.isna(yoy_latest) and yoy_latest > 5
        if c1: score += 1

        # +1 Low base but growing
        enroll_curr = _safe(r.get(f"Enroll_{T_CURR}", 0))
        og = r.get("Overall_Growth", np.nan)
        c2 = enroll_curr < 300 and not pd.isna(og) and og > 0
        if c2: score += 1

        # +1 Bright Spot (growing in red block)
        c3 = r.get("Outlier_Type", "") == "Bright Spot"
        if c3: score += 1

        # +1 Girls ratio improving
        gr_base = r.get("Girls_Ratio_Base", np.nan)
        gr_curr = r.get("Girls_Ratio_Current", np.nan)
        c4 = (not pd.isna(gr_base) and not pd.isna(gr_curr) and gr_curr > gr_base)
        if c4: score += 1

        # +1 Upper class growing
        ub = _safe(r.get(f"C11_{T_BASE}", 0)) + _safe(r.get(f"C12_{T_BASE}", 0))
        uc = _safe(r.get(f"C11_{T_CURR}", 0)) + _safe(r.get(f"C12_{T_CURR}", 0))
        c5 = uc > ub
        if c5: score += 1

        tag = "High Potential" if score >= 4 else ("Medium Potential" if score >= 2 else "Low Potential")
        risk_lvl = r.get("Risk_Level", "")
        worthy = (score >= 3 and risk_lvl != "HIGH RISK" and
                  not pd.isna(og) and og > 0)

        rows.append({
            "District":             r["district"],
            "Block":                r["block"],
            "School_ID":            r["school_id"],
            "School_Name":          r["school_name"],
            f"Total_{T_BASE}":      _safe(r.get(f"Enroll_{T_BASE}", 0)),
            f"Total_{T_CURR}":      enroll_curr,
            "Overall_Growth_pct":   og,
            "YoY_Latest":           yoy_latest,
            "Block_Zone":           r.get("Block_Zone", ""),
            "Girls_Ratio_Improving": c4,
            "Upper_Class_Growing":   c5,
            "Bright_Spot_Flag":      c3,
            "Potential_Score":       score,
            "Potential_Tag":         tag,
            "Investment_Worthy":     worthy,
        })

    result = pd.DataFrame(rows)
    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    os.makedirs(OUTPUT_PATHS["L5"], exist_ok=True)

    result.to_csv(OUTPUT_PATHS["L4"] + "13_enrollment_potential.csv", index=False)
    print(f"  Saved: {OUTPUT_PATHS['L4']}13_enrollment_potential.csv")

    worthy = result[result["Investment_Worthy"] == True].sort_values(
        "Potential_Score", ascending=False)
    worthy.to_csv(OUTPUT_PATHS["L5"] + "investment_worthy_schools.csv", index=False)
    print(f"  Saved: {OUTPUT_PATHS['L5']}investment_worthy_schools.csv")

    # District summary
    dist_sum = result.groupby("District").agg(
        High_Potential_Schools   = ("Potential_Tag",
                                    lambda x: (x == "High Potential").sum()),
        Medium_Potential_Schools = ("Potential_Tag",
                                    lambda x: (x == "Medium Potential").sum()),
        Investment_Worthy_Count  = ("Investment_Worthy", "sum"),
    ).reset_index()
    dist_sum.to_csv(OUTPUT_PATHS["L2"] + "13_district_potential.csv", index=False)
    print(f"  Saved: {OUTPUT_PATHS['L2']}13_district_potential.csv")

    hi  = int((result["Potential_Tag"] == "High Potential").sum())
    iw  = int(result["Investment_Worthy"].sum())
    print(f"\n  High Potential schools   : {hi}")
    print(f"  Investment Worthy schools: {iw}")
    print("  Top 10 Investment Worthy:")
    for _, row in worthy.head(10).iterrows():
        g = f"{row['Overall_Growth_pct']:.1f}" if not pd.isna(row['Overall_Growth_pct']) else "N/A"
        print(f"    {str(row['School_Name'])[:40]:<42} Score:{row['Potential_Score']}  {g}%  {row['District']}")

    return result


# ---------------------------------------------------------------------------
# INDEX 14: EARLY WARNING INDEX
# ---------------------------------------------------------------------------
def calc_early_warning(master):
    print("\n[Index 14] Calculating Early Warning Index...")

    ALERTS = {
        "W1":  ("ENROLLMENT CRASH",             "CRITICAL"),  # >20% fall in enrolled students
        "W2":  ("ENROLLMENT DECLINE",           "HIGH"),      # 10-20% fall in enrolled students
        "W3":  ("GIRLS ENROLLMENT CRASH",       "CRITICAL"),  # >20% fall in girls enrolled
        "W4":  ("GIRLS ENROLLMENT DROP",        "HIGH"),      # 10-20% fall in girls enrolled
        "W5":  ("CONTINUITY COLLAPSE",          "CRITICAL"),  # <60% students re-enroll next class
        "W6":  ("LOW CONTINUITY",               "HIGH"),      # 60-70% students re-enroll next class
        "W7":  ("GENDER DESERT",                "HIGH"),      # girls <20% of enrolled students
        "W8":  ("MICRO SCHOOL",                 "HIGH"),      # total enrolled < 25
        "W9":  ("4-YEAR ENROLLMENT DECLINE",    "HIGH"),      # consistent fall all 4 years
        "W10": ("MULTI-RISK ENROLLMENT",        "CRITICAL"),  # high risk score + falling enrollment
    }

    alert_rows = []
    school_alert_counts = {}

    for _, r in master.iterrows():
        sid   = r["school_id"]
        yoy   = r.get(f"Growth_{_yr_tag(PREV_YEAR)}_{T_CURR}", np.nan)
        girls_curr = _safe(r.get(f"Girls_{T_CURR}", 0))
        girls_prev = _safe(r.get(f"Girls_{_yr_tag(PREV_YEAR)}", 0))
        girls_yoy  = _pct(girls_curr, girls_prev)

        trans89 = r.get(f"IntraRatio_8_9", np.nan)  # per-school class 8→9 continuity proxy
        girls_ratio = r.get("Girls_Ratio_Current", np.nan)
        enroll_curr = _safe(r.get(f"Enroll_{T_CURR}", 0))
        trend_tag   = r.get("Trend_Tag", "")
        risk_score  = r.get("Risk_Score", 0)

        def _alert(aid, current_val, threshold, year=CURRENT_YEAR):
            alert_rows.append({
                "District":      r["district"],
                "Block":         r["block"],
                "School_ID":     sid,
                "School_Name":   r["school_name"],
                "Alert_ID":      aid,
                "Alert_Type":    ALERTS[aid][0],
                "Severity":      ALERTS[aid][1],
                "Current_Value": current_val,
                "Threshold":     threshold,
                "Year_Triggered": year,
            })
            school_alert_counts[sid] = school_alert_counts.get(sid, 0) + 1

        # W1 — Enrollment crash
        if not pd.isna(yoy) and yoy < -20:
            _alert("W1", round(yoy, 1), -20)
        # W2 — Enrollment drop
        elif not pd.isna(yoy) and -20 <= yoy < -10:
            _alert("W2", round(yoy, 1), -10)

        # W3 — Girls crash
        if not pd.isna(girls_yoy) and girls_yoy < -20:
            _alert("W3", round(girls_yoy, 1), -20)
        # W4 — Girls declining
        elif not pd.isna(girls_yoy) and -20 <= girls_yoy < -10:
            _alert("W4", round(girls_yoy, 1), -10)

        # W5 — Continuity collapse (very few students re-enroll in next class)
        if not pd.isna(trans89) and trans89 < 60:
            _alert("W5", round(trans89, 1), 60)
        # W6 — Low continuity (fewer students re-enroll in next class than expected)
        elif not pd.isna(trans89) and 60 <= trans89 < RISK_THRESHOLDS["transition_fail"]:
            _alert("W6", round(trans89, 1), RISK_THRESHOLDS["transition_fail"])

        # W7 — Gender desert
        if not pd.isna(girls_ratio) and girls_ratio < 0.20:
            _alert("W7", round(girls_ratio, 3), 0.20)

        # W8 — Micro school
        if enroll_curr > 0 and enroll_curr < RISK_THRESHOLDS["micro_school"]:
            _alert("W8", enroll_curr, RISK_THRESHOLDS["micro_school"])

        # W9 — 4-year decline
        if trend_tag == "Consistent Decline":
            _alert("W9", trend_tag, "Consistent Decline")

        # W10 — Multi-risk
        if risk_score >= 8 and not pd.isna(yoy) and yoy < 0:
            _alert("W10", int(risk_score), 8)

    alerts_df = pd.DataFrame(alert_rows)

    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    os.makedirs(OUTPUT_PATHS["L5"], exist_ok=True)

    alerts_df.to_csv(OUTPUT_PATHS["L4"] + "14_early_warning.csv", index=False)
    print(f"  Saved: {OUTPUT_PATHS['L4']}14_early_warning.csv")

    # Summary per alert type
    if not alerts_df.empty:
        summary = alerts_df.groupby(["Alert_ID", "Alert_Type", "Severity"]).agg(
            Schools_Affected  = ("School_ID", "nunique"),
            Districts_Affected = ("District", "nunique"),
            Blocks_Affected    = ("Block",    "nunique"),
        ).reset_index().sort_values("Alert_ID")
        summary.to_csv(OUTPUT_PATHS["L5"] + "early_warning_summary.csv", index=False)
        print(f"  Saved: {OUTPUT_PATHS['L5']}early_warning_summary.csv")

        critical = alerts_df[alerts_df["Severity"] == "CRITICAL"]
        critical.to_csv(OUTPUT_PATHS["L5"] + "critical_alerts.csv", index=False)
        print(f"  Saved: {OUTPUT_PATHS['L5']}critical_alerts.csv")
    else:
        summary = pd.DataFrame()

    total_alerts  = len(alerts_df)
    crit_alerts   = int((alerts_df["Severity"] == "CRITICAL").sum()) if not alerts_df.empty else 0
    multi_risk    = sum(1 for v in school_alert_counts.values() if v >= 3)

    print(f"\n  Total alerts generated : {total_alerts}")
    print(f"  CRITICAL alerts        : {crit_alerts}")
    print(f"  HIGH alerts            : {total_alerts - crit_alerts}")
    print(f"  Schools with 3+ alerts : {multi_risk}")
    if not alerts_df.empty:
        print("  Alert type distribution:")
        for _, row in summary.iterrows():
            print(f"    {row['Alert_ID']} {row['Alert_Type']:<28} {row['Severity']:<9} {row['Schools_Affected']} schools")

    return alerts_df


# ---------------------------------------------------------------------------
# INDEX 15: SEGMENT SPLIT INDEX
# ---------------------------------------------------------------------------
def calc_segment_split(master):
    print("\n[Index 15] Calculating Segment Split Index...")

    rows = []
    for _, r in master.iterrows():
        segs = {}
        for yr in YEARS:
            tag = _yr_tag(yr)
            sa = sum(_safe(r.get(f"C{c}_{tag}", 0)) for c in [8, 9, 10])
            sb = sum(_safe(r.get(f"C{c}_{tag}", 0)) for c in [11, 12])
            segs[tag] = (sa, sb)
            ratio = round(sb / (sa + 0.001) * 100, 1)
            segs[f"R_{tag}"] = ratio

        sa_base, sb_base = segs[T_BASE]
        sa_curr, sb_curr = segs[T_CURR]
        sa_g = _pct(sa_curr, sa_base)
        sb_g = _pct(sb_curr, sb_base)

        ratios = [segs[f"R_{_yr_tag(y)}"] for y in YEARS]
        ratio_trend = ("Improving"  if ratios[-1] > ratios[0]  else
                       "Declining"  if ratios[-1] < ratios[0]  else "Stable")

        # Segment tag
        if sa_g is not None and sb_g is not None:
            if sa_g > 0 and sb_g > 0:
                tag = "Both Growing"
            elif sa_g > 0 and sb_g <= 0:
                tag = "Secondary Only"
            elif sb_g > 0 and sa_g <= 0:
                tag = "Sr Secondary Only"
            else:
                tag = "Both Declining"
        elif sa_g is not None and sa_g > 0:
            tag = "Secondary Only"
        else:
            tag = "Undetermined"

        # Override with special tags
        if ratios[-1] < 30:
            tag = "Senior Collapse"
        elif all(ratios[i] >= ratios[i+1] for i in range(len(ratios)-1)) and ratios[-1] < ratios[0] - 10:
            tag = "Senior Dropout"

        rows.append({
            "District":        r["district"],
            "Block":           r["block"],
            "School_ID":       r["school_id"],
            "School_Name":     r["school_name"],
            f"SegA_{T_BASE}":  segs[T_BASE][0],
            f"SegB_{T_BASE}":  segs[T_BASE][1],
            f"SegA_{T_CURR}":  sa_curr,
            f"SegB_{T_CURR}":  sb_curr,
            "SegA_Growth_pct": sa_g,
            "SegB_Growth_pct": sb_g,
            **{f"Ratio_{_yr_tag(y)}": segs[f"R_{_yr_tag(y)}"] for y in YEARS},
            "Ratio_Trend":     ratio_trend,
            "Segment_Tag":     tag,
        })

    result = pd.DataFrame(rows)
    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    result.to_csv(OUTPUT_PATHS["L4"] + "15_segment_split.csv", index=False)
    print(f"  Saved: {OUTPUT_PATHS['L4']}15_segment_split.csv")

    # Block summary
    block_sum = result.groupby(["District", "Block"]).agg(
        Avg_Continuation_Ratio   = (f"Ratio_{T_CURR}", "mean"),
        Senior_Dropout_Schools   = ("Segment_Tag", lambda x: (x == "Senior Dropout").sum()),
        Senior_Collapse_Schools  = ("Segment_Tag", lambda x: (x == "Senior Collapse").sum()),
        Both_Growing_Schools     = ("Segment_Tag", lambda x: (x == "Both Growing").sum()),
    ).round(1).reset_index()
    block_sum.to_csv(OUTPUT_PATHS["L3"] + "15_block_segment.csv", index=False)
    print(f"  Saved: {OUTPUT_PATHS['L3']}15_block_segment.csv")

    # District summary
    dist_sum = result.groupby("District").agg(
        Avg_SegA_Growth          = ("SegA_Growth_pct", "mean"),
        Avg_SegB_Growth          = ("SegB_Growth_pct", "mean"),
        Avg_Continuation_Ratio   = (f"Ratio_{T_CURR}", "mean"),
        Senior_Collapse_Schools  = ("Segment_Tag", lambda x: (x == "Senior Collapse").sum()),
    ).round(2).reset_index()
    dist_sum["Segment_Health"] = dist_sum.apply(
        lambda r: "Strong"   if r["Avg_Continuation_Ratio"] > 60 and r["Senior_Collapse_Schools"] == 0
        else ("Weak" if r["Avg_Continuation_Ratio"] < 35 or r["Senior_Collapse_Schools"] > 3
              else "Moderate"), axis=1
    )
    dist_sum.to_csv(OUTPUT_PATHS["L2"] + "15_district_segment.csv", index=False)
    print(f"  Saved: {OUTPUT_PATHS['L2']}15_district_segment.csv")

    collapse = int((result["Segment_Tag"] == "Senior Collapse").sum())
    dropout  = int((result["Segment_Tag"] == "Senior Dropout").sum())
    avg_ratio = round(result[f"Ratio_{T_CURR}"].mean(), 1)
    worst_dist = dist_sum.nsmallest(5, "Avg_Continuation_Ratio")[["District", "Avg_Continuation_Ratio"]]

    print(f"\n  Senior Collapse schools     : {collapse}")
    print(f"  Senior Dropout schools      : {dropout}")
    print(f"  Avg Continuation Ratio (UP) : {avg_ratio}%")
    print("  Districts with worst continuation ratio:")
    for _, row in worst_dist.iterrows():
        print(f"    {row['District']:<30} {row['Avg_Continuation_Ratio']:.1f}%")

    return result


# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
def print_final_summary(growth_res, decline_res, potential_res,
                         alerts_df, segment_res):
    total_high  = len(growth_res)
    models      = int(growth_res["Model_School_Flag"].sum()) if not growth_res.empty else 0
    top_pattern = (growth_res["Growth_Pattern_Tag"].value_counts().idxmax()
                   if not growth_res.empty else "N/A")

    n_decl = len(decline_res)
    mig_n  = int((decline_res["Primary_Reason"] == "Migration/Population Shift").sum())
    gen_n  = int((decline_res["Primary_Reason"] == "Social/Gender Barrier").sum())
    tran_n = int((decline_res["Primary_Reason"] == "Higher Education Transition Issue").sum())

    hi_pot = int((potential_res["Potential_Tag"] == "High Potential").sum())
    iw     = int(potential_res["Investment_Worthy"].sum())

    crit_a = int((alerts_df["Severity"] == "CRITICAL").sum()) if not alerts_df.empty else 0
    high_a = int((alerts_df["Severity"] == "HIGH").sum())     if not alerts_df.empty else 0
    from collections import Counter
    ac = Counter()
    if not alerts_df.empty:
        for sid, cnt in pd.Series(
                alerts_df.groupby("School_ID").size().values).items():
            if cnt >= 3: ac["multi"] += 1

    collapse = int((segment_res["Segment_Tag"] == "Senior Collapse").sum())
    dropout  = int((segment_res["Segment_Tag"] == "Senior Dropout").sum())
    avg_cont = round(segment_res[f"Ratio_{T_CURR}"].mean(), 1)

    W = 52
    sep  = "+" + "="*W + "+"
    sep2 = "+" + "-"*W + "+"
    def row(txt): return "| " + txt.ljust(W-2) + " |"

    lines = [
        sep,
        row("         BATCH 3 ANALYSIS COMPLETE"),
        sep,
        row("INDEX 11 - GROWTH DRIVER"),
        row(f"  High Growth Schools : {total_high}"),
        row(f"  Model Schools       : {models}"),
        row(f"  Top Driver Pattern  : {top_pattern[:40]}"),
        sep2,
        row("INDEX 12 - DECLINE REASON"),
        row(f"  Declining Schools   : {n_decl}"),
        row(f"  Migration Cases     : {mig_n}"),
        row(f"  Gender Barrier Cases: {gen_n}"),
        row(f"  Transition Issue    : {tran_n}"),
        sep2,
        row("INDEX 13 - ENROLLMENT POTENTIAL"),
        row(f"  High Potential      : {hi_pot} schools"),
        row(f"  Investment Worthy   : {iw} schools"),
        sep2,
        row("INDEX 14 - EARLY WARNING"),
        row(f"  CRITICAL Alerts     : {crit_a}"),
        row(f"  HIGH Alerts         : {high_a}"),
        row(f"  Schools with 3+ alerts (most at risk): {ac.get('multi', 0)}"),
        sep2,
        row("INDEX 15 - SEGMENT SPLIT"),
        row(f"  Senior Collapse     : {collapse} schools"),
        row(f"  Senior Dropout      : {dropout} schools"),
        row(f"  Avg Continuation    : {avg_cont}%"),
        sep,
    ]
    print("\n" + "\n".join(lines))

    saved = [
        OUTPUT_PATHS["L4"] + "11_growth_driver.csv",
        OUTPUT_PATHS["L5"] + "model_schools.csv",
        OUTPUT_PATHS["L2"] + "11_district_growth_drivers.csv",
        OUTPUT_PATHS["L4"] + "12_decline_reason.csv",
        OUTPUT_PATHS["L3"] + "12_block_decline_reasons.csv",
        OUTPUT_PATHS["L4"] + "13_enrollment_potential.csv",
        OUTPUT_PATHS["L5"] + "investment_worthy_schools.csv",
        OUTPUT_PATHS["L2"] + "13_district_potential.csv",
        OUTPUT_PATHS["L4"] + "14_early_warning.csv",
        OUTPUT_PATHS["L5"] + "early_warning_summary.csv",
        OUTPUT_PATHS["L5"] + "critical_alerts.csv",
        OUTPUT_PATHS["L4"] + "15_segment_split.csv",
        OUTPUT_PATHS["L3"] + "15_block_segment.csv",
        OUTPUT_PATHS["L2"] + "15_district_segment.csv",
    ]
    print("\nFiles saved:")
    for f in saved:
        print(f"  {f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_long = load_data(DATA_FILE)

    (master, growth_df, gender_df, transition_df,
     block_health, district_df, risk_df, class_df,
     cluster_df) = load_inputs(df_long)

    growth_res   = calc_growth_driver(master)
    decline_res  = calc_decline_reason(master)
    potential_res = calc_enrollment_potential(master)
    alerts_df    = calc_early_warning(master)
    segment_res  = calc_segment_split(master)

    print_final_summary(growth_res, decline_res, potential_res,
                         alerts_df, segment_res)
