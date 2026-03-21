"""
Batch 1 — Core Enrollment Indices
Calculates per-school:
  1. YoY Enrollment Growth (3 pairs + overall)
  2. Enrollment Trend Tag
  3. Stability Index (CV across 4 years)
  4. Gender Enrollment Index
  5. Enrollment Continuity (class-to-class re-enrollment within each year)
  6. Class-wise Enrollment Distribution

Output: outputs/L4_SCHOOL/batch1_indices.xlsx
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (
    YEARS, YEAR_PAIRS, BASE_YEAR, CURRENT_YEAR, PREV_YEAR,
    CLASS_LIST, OUTPUT_PATHS, RISK_THRESHOLDS
)
from scripts.data_loader import load_data, create_sample_data, DATA_FILE


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _pct_change(new, old):
    """Safe percent change. Returns NaN if old is 0."""
    if old == 0:
        return np.nan
    return round((new - old) / old * 100, 2)


def _yr_tag(year_str):
    """'2022-23' → '2223'  used in column names."""
    parts = year_str.split("-")
    return parts[0][-2:] + parts[1][-2:]   # '22' + '23' = '2223'


# ---------------------------------------------------------------------------
# PIVOT: long → wide (one row per school)
# ---------------------------------------------------------------------------
def _pivot_enrollment(df):
    """
    Returns wide df with columns:
      school_id, school_name, district, block,
      School_Type, Class_Range_Label,
      Enroll_2223, Enroll_2324, Enroll_2425, Enroll_2526,
      Boys_2223 ... Boys_2526,
      Girls_2223 ... Girls_2526,
      <class_totals per year>
    """
    records = {}
    for _, row in df.iterrows():
        sid  = row["school_id"]
        yr   = row["year"]
        tag  = _yr_tag(yr)

        if sid not in records:
            records[sid] = {
                "school_id":        sid,
                "school_name":      row["school_name"],
                "district":         row["district"],
                "block":            row["block"],
                "School_Type":      row.get("School_Type", "Unknown"),
                "Class_Range_Label": row.get("Class_Range_Label", "Unknown"),
            }

        records[sid][f"Enroll_{tag}"] = row["total_enrollment"]
        records[sid][f"Boys_{tag}"]   = row["total_boys"]
        records[sid][f"Girls_{tag}"]  = row["total_girls"]

        for cls in CLASS_LIST:
            records[sid][f"Class{cls}_{tag}"] = row[f"total_{cls}"]

    wide = pd.DataFrame(list(records.values()))
    # Fill missing year-columns with 0
    for yr in YEARS:
        tag = _yr_tag(yr)
        for prefix in ["Enroll", "Boys", "Girls"] + [f"Class{c}" for c in CLASS_LIST]:
            col = f"{prefix}_{tag}"
            if col not in wide.columns:
                wide[col] = 0

    return wide


# ---------------------------------------------------------------------------
# 1. YoY GROWTH
# ---------------------------------------------------------------------------
def calc_yoy_growth(wide):
    for (y1, y2) in YEAR_PAIRS:
        t1, t2 = _yr_tag(y1), _yr_tag(y2)
        col = f"Growth_{t1}_{t2}"
        wide[col] = wide.apply(
            lambda r: _pct_change(r[f"Enroll_{t2}"], r[f"Enroll_{t1}"]), axis=1
        )

    t_base = _yr_tag(BASE_YEAR)
    t_curr = _yr_tag(CURRENT_YEAR)
    wide["Overall_Growth"] = wide.apply(
        lambda r: _pct_change(r[f"Enroll_{t_curr}"], r[f"Enroll_{t_base}"]), axis=1
    )
    return wide


# ---------------------------------------------------------------------------
# 2. TREND TAG  (based on 3 YoY growth pairs)
# ---------------------------------------------------------------------------
def _trend_tag(r):
    pairs = [(y1, y2) for (y1, y2) in YEAR_PAIRS]
    growths = []
    for (y1, y2) in pairs:
        t1, t2 = _yr_tag(y1), _yr_tag(y2)
        g = r.get(f"Growth_{t1}_{t2}", np.nan)
        growths.append(g)

    # Drop NaN
    valid = [g for g in growths if not np.isnan(g)]
    if len(valid) == 0:
        return "Insufficient Data"

    threshold = 5.0   # ±5% = stable

    all_up   = all(g > 0  for g in valid)
    all_down = all(g < 0  for g in valid)
    all_stable = all(abs(g) <= threshold for g in valid)

    if all_stable:
        return "Stable"
    if all_up:
        return "Consistent Growth"
    if all_down:
        return "Consistent Decline"

    # Peaked: up then down  (+ + -)  or  (+ -)
    if len(valid) >= 2:
        if valid[0] > 0 and valid[-1] < 0:
            return "Peaked"
        # Recovery: down then up  (- - +) or (- +)
        if valid[0] < 0 and valid[-1] > 0:
            return "Recovery"

    return "Volatile"


def calc_trend_tag(wide):
    wide["Trend_Tag"] = wide.apply(_trend_tag, axis=1)
    return wide


# ---------------------------------------------------------------------------
# 3. STABILITY INDEX  (CV across 4 years)
# ---------------------------------------------------------------------------
def calc_stability_index(wide):
    enroll_cols = [f"Enroll_{_yr_tag(y)}" for y in YEARS]
    wide["Stability_Mean"] = wide[enroll_cols].mean(axis=1).round(2)
    wide["Stability_Std"]  = wide[enroll_cols].std(axis=1).round(2)
    wide["CV_Pct"] = wide.apply(
        lambda r: round(r["Stability_Std"] / r["Stability_Mean"] * 100, 2)
        if r["Stability_Mean"] > 0 else np.nan, axis=1
    )

    cv_stable   = RISK_THRESHOLDS["cv_stable"]
    cv_volatile = RISK_THRESHOLDS["cv_volatile"]

    def _stability_label(cv):
        if np.isnan(cv):
            return "Unknown"
        if cv <= cv_stable:
            return "Stable"
        if cv <= cv_volatile:
            return "Moderate"
        return "Volatile"

    wide["Stability_Label"] = wide["CV_Pct"].apply(_stability_label)
    return wide


# ---------------------------------------------------------------------------
# 4. GENDER INDEX
# ---------------------------------------------------------------------------
def calc_gender_index(wide):
    t_base = _yr_tag(BASE_YEAR)
    t_curr = _yr_tag(CURRENT_YEAR)

    wide["Girls_Growth_Pct"] = wide.apply(
        lambda r: _pct_change(r[f"Girls_{t_curr}"], r[f"Girls_{t_base}"]), axis=1
    )
    wide["Boys_Growth_Pct"] = wide.apply(
        lambda r: _pct_change(r[f"Boys_{t_curr}"], r[f"Boys_{t_base}"]), axis=1
    )
    wide["Gender_Gap_Pct"] = (wide["Girls_Growth_Pct"] - wide["Boys_Growth_Pct"]).round(2)

    # Girls ratio in current year
    curr_enroll = f"Enroll_{t_curr}"
    curr_girls  = f"Girls_{t_curr}"
    wide["Girls_Ratio_Current"] = wide.apply(
        lambda r: round(r[curr_girls] / r[curr_enroll], 3)
        if r[curr_enroll] > 0 else np.nan, axis=1
    )
    wide["Girls_Imbalance_Flag"] = wide["Girls_Ratio_Current"].apply(
        lambda x: "Yes" if (not np.isnan(x) and x < RISK_THRESHOLDS["girls_imbalance"]) else "No"
    )
    return wide


# ---------------------------------------------------------------------------
# 5. ENROLLMENT CONTINUITY INDEX  (Class N+1 / Class N within same year)
#    Measures: are students re-enrolling in the next class?
# ---------------------------------------------------------------------------
def calc_transition_index(df_long, wide):
    """
    For each year compute class-to-class enrollment continuity ratios.
    Continuity_8_9 = Class9 enrolled / Class8 enrolled * 100
    Primary analysis = CURRENT_YEAR.
    """
    transitions = {yr: {} for yr in YEARS}

    for yr in YEARS:
        yr_df = df_long[df_long["year"] == yr]
        if yr_df.empty:
            continue
        tag = _yr_tag(yr)
        for i in range(len(CLASS_LIST) - 1):
            c_from = CLASS_LIST[i]
            c_to   = CLASS_LIST[i + 1]
            total_from = yr_df[f"total_{c_from}"].sum()
            total_to   = yr_df[f"total_{c_to}"].sum()
            ratio = round(total_to / total_from * 100, 2) if total_from > 0 else np.nan
            transitions[yr][f"Continuity_{c_from}_{c_to}_{tag}"] = ratio

    # Merge continuity cols into wide (aggregate level — same value for all schools)
    for yr, trans_dict in transitions.items():
        for col, val in trans_dict.items():
            wide[col] = val

    # Current year primary continuity summary
    t_curr = _yr_tag(CURRENT_YEAR)
    cont_cols_curr = [f"Continuity_{CLASS_LIST[i]}_{CLASS_LIST[i+1]}_{t_curr}"
                      for i in range(len(CLASS_LIST) - 1)]
    available = [c for c in cont_cols_curr if c in wide.columns]
    if available:
        wide["Avg_Continuity_Current"] = wide[available].mean(axis=1).round(2)

    return wide


# ---------------------------------------------------------------------------
# 6. CLASS-WISE ENROLLMENT DISTRIBUTION
#    Shows how students are distributed across classes.
#    Enrollment_Gap = (Class8 - Class12) / Class8 * 100
#    Wide gap = students concentrated in lower classes
#    Narrow gap = enrollment is balanced across all classes
# ---------------------------------------------------------------------------
def calc_dropout_proxy(wide):
    t_curr = _yr_tag(CURRENT_YEAR)
    t_base = _yr_tag(BASE_YEAR)

    def _enroll_gap(cl8, cl12):
        if cl8 == 0:
            return np.nan
        return round((cl8 - cl12) / cl8 * 100, 2)

    wide["Enroll_Gap_Current"] = wide.apply(
        lambda r: _enroll_gap(r[f"Class8_{t_curr}"], r[f"Class12_{t_curr}"]), axis=1
    )
    wide["Enroll_Gap_Base"] = wide.apply(
        lambda r: _enroll_gap(r[f"Class8_{t_base}"], r[f"Class12_{t_base}"]), axis=1
    )
    wide["Enroll_Gap_Change"] = (wide["Enroll_Gap_Current"] - wide["Enroll_Gap_Base"]).round(2)
    wide["Enroll_Pattern_Tag"] = wide["Enroll_Gap_Current"].apply(
        lambda x: "Wide Spread"  if (not np.isnan(x) and x >= RISK_THRESHOLDS["dropout_high"])
        else ("Moderate" if (not np.isnan(x) and x >= 15) else "Narrow Spread")
    )
    return wide


# ---------------------------------------------------------------------------
# INDEX 1B: SCHOOL TYPE ENROLLMENT TREND
# ---------------------------------------------------------------------------
def calc_schooltype_trend(wide, df_long):
    """
    Group schools by School_Type and calculate:
    - Enrollment per year (total + average)
    - Overall growth 2022-23 to 2025-26
    - Schools growing vs declining
    - Girls ratio, Continuity 8->9 per type
    Output: outputs/L1_STATE/01b_schooltype_trend.csv
    """
    print("\n[Index 1B] School Type Enrollment Trend...")

    t_base = _yr_tag(BASE_YEAR)
    t_curr = _yr_tag(CURRENT_YEAR)
    t_prev = _yr_tag(PREV_YEAR)

    rows = []
    for stype, grp in wide.groupby("School_Type"):
        n = len(grp)
        clr = grp["Class_Range_Label"].iloc[0] if len(grp) > 0 else "?"

        avg_enroll, total_enroll = {}, {}
        for yr in YEARS:
            tag = _yr_tag(yr)
            col = f"Enroll_{tag}"
            avg_enroll[yr]   = round(grp[col].mean(), 1) if col in grp.columns else 0
            total_enroll[yr] = int(grp[col].sum())       if col in grp.columns else 0

        base = avg_enroll[BASE_YEAR]
        curr = avg_enroll[CURRENT_YEAR]
        prev = avg_enroll[PREV_YEAR]

        overall_g = round((curr - base) / base * 100, 1) if base > 0 else np.nan
        yoy_latest = round((curr - prev) / prev * 100, 1) if prev > 0 else np.nan

        growing_n  = int((grp["Trend_Tag"].isin(["Consistent Growth", "Recovery"])).sum())
        declining_n = int((grp["Trend_Tag"].isin(["Consistent Decline", "Peaked"])).sum())

        girls_col = f"Girls_{t_curr}"
        enroll_col = f"Enroll_{t_curr}"
        avg_girls_ratio = round(
            (grp[girls_col].sum() / grp[enroll_col].sum()) if (
                girls_col in grp.columns and grp[enroll_col].sum() > 0) else np.nan, 3
        )

        # Per-type continuity 8→9: from df_long (class-level data)
        type_schools = set(grp["school_id"].tolist())
        type_long = df_long[(df_long["school_id"].isin(type_schools)) &
                            (df_long["year"] == CURRENT_YEAR)]
        c8_sum = int(type_long["total_8"].sum())
        c9_sum = int(type_long["total_9"].sum())
        avg_cont_89 = round(c9_sum / c8_sum * 100, 1) if c8_sum > 0 else np.nan

        # Health tag
        g_pct = growing_n / n * 100 if n > 0 else 0
        d_pct = declining_n / n * 100 if n > 0 else 0
        if g_pct > 60 and not np.isnan(overall_g) and overall_g > 10:
            health = "Thriving"
        elif g_pct > 50:
            health = "Growing"
        elif d_pct > 70 or (not np.isnan(overall_g) and overall_g < -15):
            health = "Critical"
        elif d_pct > 50:
            health = "Declining"
        else:
            health = "Mixed"

        rows.append({
            "School_Type":          stype,
            "Class_Range":          clr,
            "Total_Schools":        n,
            f"Avg_Enroll_{_yr_tag(YEARS[0])}": avg_enroll[YEARS[0]],
            f"Avg_Enroll_{_yr_tag(YEARS[1])}": avg_enroll[YEARS[1]],
            f"Avg_Enroll_{_yr_tag(YEARS[2])}": avg_enroll[YEARS[2]],
            f"Avg_Enroll_{_yr_tag(YEARS[3])}": avg_enroll[YEARS[3]],
            f"Total_Enroll_{t_base}": total_enroll[BASE_YEAR],
            f"Total_Enroll_{t_curr}": total_enroll[CURRENT_YEAR],
            "Overall_Growth_pct":   overall_g,
            "YoY_Latest_pct":       yoy_latest,
            "Growing_Schools":      growing_n,
            "Declining_Schools":    declining_n,
            "Growing_pct":          round(g_pct, 1),
            "Declining_pct":        round(d_pct, 1),
            "Avg_Girls_Ratio":      avg_girls_ratio,
            "Avg_Continuity_8_9":   avg_cont_89,
            "Type_Health_Tag":      health,
        })

    result = pd.DataFrame(rows).sort_values("Overall_Growth_pct", ascending=False)

    os.makedirs(OUTPUT_PATHS["L1"], exist_ok=True)
    out = OUTPUT_PATHS["L1"] + "01b_schooltype_trend.csv"
    result.to_csv(out, index=False)

    # Print summary table
    W = 65
    sep  = "+" + "="*W + "+"
    sep2 = "+" + "-"*W + "+"
    def row(t): return "| " + t.ljust(W-2) + " |"

    print("\n" + sep)
    print(row("       SCHOOL TYPE ENROLLMENT ANALYSIS"))
    print(sep)
    hdr = f"  {'School Type':<24} {'Range':<7} {'Schools':>7}  {'Growth%':>8}  Health"
    print(row(hdr))
    print(sep2)
    for _, r in result.iterrows():
        g = f"{r['Overall_Growth_pct']:+.1f}%" if not pd.isna(r['Overall_Growth_pct']) else "  N/A "
        line = f"  {r['School_Type']:<24} {r['Class_Range']:<7} {int(r['Total_Schools']):>7}  {g:>8}  {r['Type_Health_Tag']}"
        print(row(line))
    print(sep)

    return result


# ---------------------------------------------------------------------------
# SUMMARY PRINT
# ---------------------------------------------------------------------------
def print_summary(wide):
    print(f"\n{'='*60}")
    print("BATCH 1 — ENROLLMENT INDICES SUMMARY")
    print(f"4-year analysis period: {BASE_YEAR} to {CURRENT_YEAR}")
    print('='*60)
    print(f"Total Schools Analyzed  : {len(wide)}")

    t_base = _yr_tag(BASE_YEAR)
    t_curr = _yr_tag(CURRENT_YEAR)
    print(f"\nEnrollment (State Total):")
    print(f"  {BASE_YEAR}   : {wide[f'Enroll_{t_base}'].sum():,}")
    for y1, y2 in YEAR_PAIRS:
        t2 = _yr_tag(y2)
        print(f"  {y2}   : {wide[f'Enroll_{t2}'].sum():,}")

    print(f"\nYoY Growth (Median %):")
    for y1, y2 in YEAR_PAIRS:
        t1, t2 = _yr_tag(y1), _yr_tag(y2)
        col = f"Growth_{t1}_{t2}"
        print(f"  {y1} -> {y2} : {wide[col].median():.1f}%")
    print(f"  Overall ({BASE_YEAR} -> {CURRENT_YEAR}) : {wide['Overall_Growth'].median():.1f}%")

    print(f"\nTrend Distribution:")
    for tag, cnt in wide["Trend_Tag"].value_counts().items():
        print(f"  {tag:22s}: {cnt}")

    print(f"\nStability Distribution:")
    for label, cnt in wide["Stability_Label"].value_counts().items():
        print(f"  {label:22s}: {cnt}")

    print(f"\nGender (Girls Imbalance Flag = Yes) : {(wide['Girls_Imbalance_Flag']=='Yes').sum()}")
    print(f"Low Continuity schools              : {(wide.get('Low_Continuity_Flag', pd.Series())=='Yes').sum()}")
    print(f"Wide Enroll Spread schools          : {(wide.get('Enroll_Pattern_Tag', pd.Series())=='Wide Spread').sum()}")
    print('='*60)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print("Sample data not found — creating...")
        create_sample_data()

    df_long = load_data(DATA_FILE)

    print("\nPivoting to wide format...")
    wide = _pivot_enrollment(df_long)

    print("Calculating YoY Growth...")
    wide = calc_yoy_growth(wide)

    print("Calculating Trend Tags...")
    wide = calc_trend_tag(wide)

    print("Calculating Stability Index...")
    wide = calc_stability_index(wide)

    print("Calculating Gender Index...")
    wide = calc_gender_index(wide)

    print("Calculating Transition Index...")
    wide = calc_transition_index(df_long, wide)

    print("Calculating Class Enrollment Distribution...")
    wide = calc_dropout_proxy(wide)

    # Add flags needed by batch2
    wide["Low_Continuity_Flag"] = wide.get("Avg_Continuity_Current", pd.Series(dtype=float)).apply(
        lambda x: "Yes" if (not pd.isna(x) and x < RISK_THRESHOLDS["transition_fail"]) else "No"
    )
    wide["Weak_Continuity_Flag"] = wide.get("Avg_Continuity_Current", pd.Series(dtype=float)).apply(
        lambda x: "Yes" if (not pd.isna(x) and
                            RISK_THRESHOLDS["transition_fail"] <= x < RISK_THRESHOLDS["transition_weak"]) else "No"
    )
    wide["Girls_Declining_Flag"] = wide["Girls_Growth_Pct"].apply(
        lambda x: "Yes" if (not pd.isna(x) and x < 0) else "No"
    )

    print("Calculating School Type Enrollment Trend...")
    schooltype_df = calc_schooltype_trend(wide, df_long)

    os.makedirs(OUTPUT_PATHS["L4"], exist_ok=True)
    os.makedirs(OUTPUT_PATHS["L1"], exist_ok=True)

    # Save master xlsx
    out_path = OUTPUT_PATHS["L4"] + "batch1_indices.xlsx"
    wide.to_excel(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(wide)} schools x {len(wide.columns)} columns)")

    # ---- Save 5 separate CSVs for batch2 ----
    id_cols = ["school_id", "school_name", "district", "block", "School_Type", "Class_Range_Label"]
    enroll_cols = [f"Enroll_{_yr_tag(y)}" for y in YEARS]
    growth_cols = [f"Growth_{_yr_tag(y1)}_{_yr_tag(y2)}" for y1, y2 in YEAR_PAIRS] + ["Overall_Growth", "Trend_Tag"]

    boys_cols  = [f"Boys_{_yr_tag(y)}"  for y in YEARS]
    girls_cols = [f"Girls_{_yr_tag(y)}" for y in YEARS]

    trans_cols = [c for c in wide.columns if c.startswith("Continuity_")]
    class_cols = [c for c in wide.columns if c.startswith("Class") and "_" in c]

    csv1 = wide[id_cols + enroll_cols + growth_cols].copy()
    csv1.to_csv(OUTPUT_PATHS["L4"] + "01_growth_decline.csv", index=False)

    csv2 = wide[id_cols + ["Stability_Mean", "Stability_Std", "CV_Pct", "Stability_Label"]].copy()
    csv2.to_csv(OUTPUT_PATHS["L4"] + "02_stability_index.csv", index=False)

    csv3 = wide[id_cols + boys_cols + girls_cols +
                ["Girls_Growth_Pct", "Boys_Growth_Pct", "Gender_Gap_Pct",
                 "Girls_Ratio_Current", "Girls_Imbalance_Flag", "Girls_Declining_Flag"]].copy()
    csv3.to_csv(OUTPUT_PATHS["L4"] + "03_gender_equity.csv", index=False)

    cont_avg_col = ["Avg_Continuity_Current"] if "Avg_Continuity_Current" in wide.columns else []
    csv4 = wide[id_cols + trans_cols + cont_avg_col +
                ["Low_Continuity_Flag", "Weak_Continuity_Flag"]].copy()
    csv4.to_csv(OUTPUT_PATHS["L4"] + "04_continuity_index.csv", index=False)

    csv5 = wide[id_cols + ["Enroll_Gap_Current", "Enroll_Gap_Base",
                            "Enroll_Gap_Change", "Enroll_Pattern_Tag"]].copy()
    csv5.to_csv(OUTPUT_PATHS["L4"] + "05_class_distribution.csv", index=False)

    print("Saved 5 CSVs (01-05) to", OUTPUT_PATHS["L4"])
    print("  01_growth_decline.csv        - Enrollment numbers + YoY growth")
    print("  02_stability_index.csv       - Enrollment stability (CV)")
    print("  03_gender_equity.csv         - Boys/Girls enrollment trends")
    print("  04_continuity_index.csv      - Class-to-class enrollment continuity")
    print("  05_class_distribution.csv   - Class 8 vs 12 enrollment spread")

    print_summary(wide)
