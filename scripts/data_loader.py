import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import DATA_FILE, YEARS, CLASS_LIST, GOVT_FILES

YEAR_TAGS = ["2022", "2023", "2024", "2025"]   # first part of "2022-23" etc.

# ---------------------------------------------------------------------------
# STANDARD OUTPUT COLUMNS (always returned by load_data)
# ---------------------------------------------------------------------------
STD_COLS = [
    "school_id", "school_name", "district", "block", "year",
    "school_category",
    "boys_8",  "girls_8",
    "boys_9",  "girls_9",
    "boys_10", "girls_10",
    "boys_11", "girls_11",
    "boys_12", "girls_12",
    "total_8", "total_9", "total_10", "total_11", "total_12",
    "total_boys", "total_girls", "total_enrollment",
]

# Short readable labels for school categories
CATEGORY_LABELS = {
    "Secondary + Higher Secondary (9\x9612)"       : "Sec+HSec (9-12)",
    "Upper Primary to Higher Secondary (6\x9612)"  : "UP+HSec (6-12)",
    "Upper Primary + Secondary (6\x9610)"          : "UP+Sec (6-10)",
    "Upper Primary to Secondary (6\x9610)"         : "UP+Sec (6-10)",
    "Primary to Higher Secondary (1\x9612)"        : "Pri+HSec (1-12)",
    "Secondary only (9\x9610)"                     : "Sec only (9-10)",
}

# Maps normalized school_category substring → (School_Type, Lowest_Class, Highest_Class)
# school_category strings use \x96 (Windows-1252 en-dash) as separator; we normalise to "-"
SCHOOL_TYPE_MAP = [
    # MORE SPECIFIC keys first (longer keys before shorter substrings)
    ("upper primary to higher secondary",    "Middle_to_Higher",      6,  12),
    ("upper primary + secondary",            "Middle_to_Secondary",   6,  10),
    ("upper primary to secondary",           "Middle_to_Secondary",   6,  10),
    ("primary to higher secondary",          "Full_School",           1,  12),  # after "upper primary..."
    ("secondary + higher secondary",         "Senior_Secondary",      9,  12),
    ("secondary only",                       "Secondary_Only",        9,  10),
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _norm_col(c):
    """Lowercase + strip. Keep hyphens so '2022-23' stays intact."""
    return str(c).strip().lower().replace(" ", "_")


def _safe_int(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _build_standard_row(school_id, school_name, district, block, year,
                         boys=None, girls=None, totals=None):
    boys   = boys   or {c: 0 for c in CLASS_LIST}
    girls  = girls  or {c: 0 for c in CLASS_LIST}
    totals = totals or {c: boys.get(c, 0) + girls.get(c, 0) for c in CLASS_LIST}

    row = {
        "school_id":   school_id,
        "school_name": school_name,
        "district":    district,
        "block":       block,
        "year":        str(year),
    }
    for c in CLASS_LIST:
        row[f"boys_{c}"]  = boys.get(c, 0)
        row[f"girls_{c}"] = girls.get(c, 0)
    for c in CLASS_LIST:
        row[f"total_{c}"] = totals.get(c, 0)

    row["total_boys"]       = sum(boys.get(c, 0)  for c in CLASS_LIST)
    row["total_girls"]      = sum(girls.get(c, 0) for c in CLASS_LIST)
    row["total_enrollment"] = sum(totals.get(c, 0) for c in CLASS_LIST)
    return row


# ---------------------------------------------------------------------------
# SCHOOL TYPE DETECTION
# ---------------------------------------------------------------------------
def detect_school_type(row):
    """
    Determine school type from school_category string (primary source)
    or from which classes have non-zero enrollment (fallback).

    Returns dict with:
      School_Type, Class_Range_Label, Lowest_Class, Highest_Class,
      Has_Primary, Has_Middle, Has_Secondary, Has_Higher_Secondary
    """
    # --- Primary: use school_category ---
    cat = str(row.get("school_category", "") or "").strip()
    # Normalise: replace Windows en-dash \x96 / Unicode dashes with plain "-"
    cat_norm = cat.lower().replace("\x96", "-").replace("\u2013", "-").replace("\u2014", "-")

    for key, stype, lo, hi in SCHOOL_TYPE_MAP:
        if key in cat_norm:
            return {
                "School_Type":           stype,
                "Class_Range_Label":     f"{lo}-{hi}",
                "Lowest_Class":          lo,
                "Highest_Class":         hi,
                "Has_Primary":           lo <= 5,
                "Has_Middle":            lo <= 8 and hi >= 6,
                "Has_Secondary":         lo <= 10 and hi >= 9,
                "Has_Higher_Secondary":  hi >= 11,
            }

    # --- Fallback: enrollment-based detection (Class 8-12 data only) ---
    present = [c for c in CLASS_LIST if int(row.get(f"total_{c}", 0) or 0) > 0]
    if not present:
        return {
            "School_Type": "Unknown", "Class_Range_Label": "Unknown",
            "Lowest_Class": 0, "Highest_Class": 0,
            "Has_Primary": False, "Has_Middle": False,
            "Has_Secondary": False, "Has_Higher_Secondary": False,
        }

    lo, hi = min(present), max(present)
    if lo == 8 and hi == 12:   stype = "High_School"
    elif lo >= 9 and hi == 12: stype = "Senior_Secondary"
    elif hi <= 10:             stype = "Secondary_Only"
    elif hi >= 11:             stype = "Senior_Secondary"
    else:                      stype = "Other"

    return {
        "School_Type":          stype,
        "Class_Range_Label":    f"{lo}-{hi}",
        "Lowest_Class":         lo,
        "Highest_Class":        hi,
        "Has_Primary":          False,   # can't determine from class 8-12 data
        "Has_Middle":           lo <= 8,
        "Has_Secondary":        hi >= 9,
        "Has_Higher_Secondary": hi >= 11,
    }


def apply_school_types(df):
    """
    Apply detect_school_type to every row and attach result columns.
    Safe to call on any df that has school_category and/or total_8..total_12.
    """
    type_cols = ["School_Type", "Class_Range_Label", "Lowest_Class", "Highest_Class",
                 "Has_Primary", "Has_Middle", "Has_Secondary", "Has_Higher_Secondary"]

    results = [detect_school_type(row) for _, row in df.iterrows()]
    type_df  = pd.DataFrame(results, index=df.index)
    for col in type_cols:
        df[col] = type_df[col]
    return df


def _print_school_type_dist(df):
    """Print school type distribution (one row per school, using any year)."""
    uniq = df.drop_duplicates(subset="school_id")
    total = len(uniq)
    dist  = uniq["School_Type"].value_counts()
    clr   = uniq.groupby("School_Type")["Class_Range_Label"].first()
    print("\nSchool Type Distribution:")
    for stype, cnt in dist.items():
        pct = cnt / total * 100
        rng = clr.get(stype, "?")
        print(f"  {stype:<25} ({rng:<5}): {cnt:4d} schools  ({pct:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# FORMAT DETECTION
# ---------------------------------------------------------------------------
def detect_format(filepath):
    xl = pd.ExcelFile(filepath)
    sheets = xl.sheet_names

    # FORMAT C: multiple sheets named like years  e.g. "2022-23"
    year_like = [s for s in sheets if any(t in str(s) for t in YEAR_TAGS)]
    if len(year_like) >= 2:
        print(f"[detect_format] FORMAT C detected — multiple year sheets: {year_like}")
        return "C"

    df = pd.read_excel(filepath, sheet_name=sheets[0], nrows=5)
    cols = [_norm_col(c) for c in df.columns]

    # FORMAT E: combined total only — total_2022-23 style
    total_year_cols = [c for c in cols if c.startswith("total_") and
                       any(t in c for t in YEAR_TAGS)]
    if total_year_cols and not any("boys" in c or "girls" in c for c in cols):
        print(f"[detect_format] FORMAT E detected — combined totals: {total_year_cols}")
        return "E"

    # FORMAT B: wide — year-suffixed columns like boys_8_2022-23
    wide_cols = [c for c in cols if any(t in c for t in YEAR_TAGS)
                 and ("boys" in c or "girls" in c or "class" in c)]
    if wide_cols:
        print(f"[detect_format] FORMAT B detected — wide year columns: {wide_cols[:4]} ...")
        return "B"

    # FORMAT D: total only (class_8 … class_12, no boys/girls)
    has_class_cols = any(f"class_{c}" in cols for c in CLASS_LIST)
    has_gender     = any("boys" in c or "girls" in c for c in cols)
    has_year_col   = "year" in cols

    if has_class_cols and not has_gender:
        print("[detect_format] FORMAT D detected — class totals, no gender split")
        return "D"

    # FORMAT A: long format with boys/girls per class
    if has_gender and has_year_col:
        print("[detect_format] FORMAT A detected — long format with gender split")
        return "A"

    print("[detect_format] FORMAT A assumed (fallback)")
    return "A"


# ---------------------------------------------------------------------------
# LOADERS PER FORMAT
# ---------------------------------------------------------------------------
def _load_format_a(filepath):
    df = pd.read_excel(filepath)
    df.columns = [_norm_col(c) for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        boys  = {c: int(r.get(f"boys_{c}",  0) or 0) for c in CLASS_LIST}
        girls = {c: int(r.get(f"girls_{c}", 0) or 0) for c in CLASS_LIST}
        row = _build_standard_row(
            r.get("school_id", ""), r.get("school_name", ""),
            r.get("district", ""),  r.get("block", ""),
            str(r.get("year", "")),
            boys=boys, girls=girls
        )
        # Carry through extra string columns if present in source
        row["school_category"] = r.get("school_category", "")
        rows.append(row)
    return pd.DataFrame(rows)


def _load_format_b(filepath):
    df = pd.read_excel(filepath)
    df.columns = [_norm_col(c) for c in df.columns]
    years_found = []
    for y in YEARS:
        tag = y.replace("-", "_")   # "2022-23" → "2022_23" in col names
        if any(tag in c for c in df.columns):
            years_found.append(y)
    rows = []
    for _, r in df.iterrows():
        for y in years_found:
            tag = y.replace("-", "_")
            boys  = {c: int(r.get(f"boys_{c}_{tag}",  0) or 0) for c in CLASS_LIST}
            girls = {c: int(r.get(f"girls_{c}_{tag}", 0) or 0) for c in CLASS_LIST}
            rows.append(_build_standard_row(
                r.get("school_id", ""), r.get("school_name", ""),
                r.get("district", ""),  r.get("block", ""),
                y, boys=boys, girls=girls
            ))
    return pd.DataFrame(rows)


def _load_format_c(filepath):
    xl = pd.ExcelFile(filepath)
    rows = []
    for sheet in xl.sheet_names:
        year = None
        for y in YEARS:
            if y in str(sheet) or y[:4] in str(sheet):
                year = y
                break
        if year is None:
            continue
        df = pd.read_excel(filepath, sheet_name=sheet)
        df.columns = [_norm_col(c) for c in df.columns]
        for _, r in df.iterrows():
            boys  = {c: int(r.get(f"boys_{c}",  0) or 0) for c in CLASS_LIST}
            girls = {c: int(r.get(f"girls_{c}", 0) or 0) for c in CLASS_LIST}
            rows.append(_build_standard_row(
                r.get("school_id", ""), r.get("school_name", ""),
                r.get("district", ""),  r.get("block", ""),
                year, boys=boys, girls=girls
            ))
    return pd.DataFrame(rows)


def _load_format_d(filepath):
    df = pd.read_excel(filepath)
    df.columns = [_norm_col(c) for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        totals = {c: int(r.get(f"class_{c}", 0) or 0) for c in CLASS_LIST}
        rows.append(_build_standard_row(
            r.get("school_id", ""), r.get("school_name", ""),
            r.get("district", ""),  r.get("block", ""),
            str(r.get("year", "")),
            totals=totals
        ))
    return pd.DataFrame(rows)


def _load_format_e(filepath):
    df = pd.read_excel(filepath)
    df.columns = [_norm_col(c) for c in df.columns]
    year_cols = {}
    for y in YEARS:
        tag = y.replace("-", "_")
        for c in df.columns:
            if c.startswith("total_") and tag in c:
                year_cols[y] = c
                break
    rows = []
    for _, r in df.iterrows():
        for year, col in year_cols.items():
            total = int(r.get(col, 0) or 0)
            row = _build_standard_row(
                r.get("school_id", ""), r.get("school_name", ""),
                r.get("district", ""),  r.get("block", ""),
                year,
                totals={c: 0 for c in CLASS_LIST}
            )
            row["total_enrollment"] = total
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN load_data
# ---------------------------------------------------------------------------
def load_data(filepath=DATA_FILE):
    print(f"\n{'='*55}")
    print(f"Loading: {filepath}")
    print('='*55)

    fmt = detect_format(filepath)

    loaders = {
        "A": _load_format_a,
        "B": _load_format_b,
        "C": _load_format_c,
        "D": _load_format_d,
        "E": _load_format_e,
    }
    df = loaders[fmt](filepath)

    for col in STD_COLS:
        if col not in df.columns:
            df[col] = 0

    df = df[STD_COLS].copy()

    # Numeric coercion — exclude non-numeric identity columns
    skip = {"school_id", "school_name", "district", "block", "year", "school_category"}
    num_cols = [c for c in STD_COLS if c not in skip]
    for col in num_cols:
        df[col] = _safe_int(df[col])

    # Add school type columns
    df = apply_school_types(df)
    _print_school_type_dist(df)

    print(f"[load_data] Loaded {len(df)} records in FORMAT {fmt}")
    return df


# ---------------------------------------------------------------------------
# VALIDATE
# ---------------------------------------------------------------------------
def validate_data(df):
    print(f"\n{'='*55}")
    print("DATA VALIDATION REPORT")
    print('='*55)

    report = {}

    total_schools = df["school_id"].nunique()
    report["total_schools"] = total_schools
    print(f"1. Total Schools        : {total_schools}")

    years = sorted(df["year"].unique().tolist())
    report["years"] = years
    print(f"2. Years Found          : {years}")

    districts = sorted(df["district"].unique().tolist())
    report["districts"] = districts
    print(f"3. Districts Found      : {len(districts)} -> {districts}")

    blocks = sorted(df["block"].unique().tolist())
    report["blocks"] = blocks
    print(f"4. Blocks Found         : {len(blocks)} -> {blocks}")

    missing = df.isnull().sum().sum()
    report["missing_values"] = int(missing)
    print(f"5. Missing Values       : {missing}")

    non_numeric = {"school_id", "school_name", "district", "block", "year",
                   "school_category", "School_Type", "Class_Range_Label"}
    num_cols = [c for c in df.columns if c not in non_numeric and df[c].dtype != object]
    neg_count = int((df[num_cols] < 0).any(axis=1).sum())
    report["negative_value_rows"] = neg_count
    if neg_count:
        print(f"6. Negative Values      : ERROR — {neg_count} rows have negative values!")
    else:
        print(f"6. Negative Values      : OK — none found")

    zero_enroll = int((df["total_enrollment"] == 0).sum())
    report["zero_enrollment_schools"] = zero_enroll
    if zero_enroll:
        print(f"7. Zero Enrollment      : WARNING — {zero_enroll} school-year records have 0 enrollment")
    else:
        print(f"7. Zero Enrollment      : OK — none found")

    gender_cols = [c for c in df.columns if c.startswith("boys_") or c.startswith("girls_")]
    has_gender = int(df[gender_cols].sum().sum()) > 0
    report["gender_data_available"] = has_gender
    print(f"8. Gender Data          : {'Yes' if has_gender else 'No'}")

    class_cols = [f"total_{c}" for c in CLASS_LIST]
    has_classwise = int(df[class_cols].sum().sum()) > 0
    report["classwise_data_available"] = has_classwise
    print(f"9. Class-wise Data      : {'Yes' if has_classwise else 'No'}")

    print('='*55)
    return report


# ---------------------------------------------------------------------------
# CREATE SAMPLE DATA  (4 years: 2022-23 to 2025-26)
# ---------------------------------------------------------------------------
def create_sample_data():
    import random
    random.seed(42)
    np.random.seed(42)

    districts = ["AGRA", "MATHURA", "FIROZABAD"]
    blocks_map = {
        "AGRA":      ["ACCHNERA", "BICHPURI", "ETMADPUR"],
        "MATHURA":   ["BALDEO",   "CHHATA",   "FARAH"],
        "FIROZABAD": ["JASRANA",  "SHIKOHABAD", "TUNDLA"],
    }

    sample_years = ["2022-23", "2023-24", "2024-25", "2025-26"]

    school_profiles = []
    sid = 1
    for dist in districts:
        for block in blocks_map[dist]:
            for i in range(5):
                profile = random.choice(["growing", "declining", "stable"])
                risk    = random.choice(["high", "medium", "low"])
                school_profiles.append({
                    "school_id":   f"SCH{sid:04d}",
                    "school_name": f"{block} GOVT SCHOOL {i+1}",
                    "district":    dist,
                    "block":       block,
                    "profile":     profile,
                    "risk":        risk,
                })
                sid += 1

    rows = []
    for sp in school_profiles:
        base = {
            "growing":   random.randint(30, 80),
            "declining": random.randint(60, 120),
            "stable":    random.randint(40, 90),
        }[sp["profile"]]

        risk_mult = {"high": 0.5, "medium": 0.85, "low": 1.2}[sp["risk"]]
        base = int(base * risk_mult)

        for yr_idx, year in enumerate(sample_years):
            growth_factor = {
                "growing":   1.0 + yr_idx * random.uniform(0.05, 0.20),
                "declining": 1.0 - yr_idx * random.uniform(0.05, 0.15),
                "stable":    1.0 + yr_idx * random.uniform(-0.03, 0.03),
            }[sp["profile"]]

            total_enroll = max(5, int(base * growth_factor))

            class_weights = [0.30, 0.25, 0.22, 0.13, 0.10]
            class_totals  = {}
            remaining = total_enroll
            for idx, cls in enumerate(CLASS_LIST):
                if idx < 4:
                    ct = max(0, int(total_enroll * class_weights[idx] + random.randint(-3, 3)))
                else:
                    ct = max(0, remaining)
                class_totals[cls] = ct
                remaining -= ct

            girls_ratio = random.uniform(0.35, 0.55)
            boys_d, girls_d = {}, {}
            for cls in CLASS_LIST:
                g = int(class_totals[cls] * girls_ratio)
                boys_d[cls]  = class_totals[cls] - g
                girls_d[cls] = g

            row = {
                "school_id":   sp["school_id"],
                "school_name": sp["school_name"],
                "district":    sp["district"],
                "block":       sp["block"],
                "year":        year,
            }
            for cls in CLASS_LIST:
                row[f"boys_{cls}"]  = boys_d[cls]
                row[f"girls_{cls}"] = girls_d[cls]
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_excel("data/enrollment_raw.xlsx", index=False)
    print("Sample data created successfully")
    print(f"  -> {len(school_profiles)} schools x 4 years = {len(df)} rows")
    print(f"  -> Years: {sample_years}")
    print(f"  -> Saved to: data/enrollment_raw.xlsx")
    return df


# ---------------------------------------------------------------------------
# LOAD REAL GOVT DATA  (4 matched xlsx files -> standard format)
# ---------------------------------------------------------------------------
def load_govt_data(save_path=DATA_FILE):
    """
    Reads the 4 year-wise Govt_Matched xlsx files,
    maps columns to standard format, merges, and saves to save_path.
    Returns standard long-format dataframe.
    """
    print(f"\n{'='*55}")
    print("Loading REAL GOVT DATA (4-year matched files)")
    print('='*55)

    col_map = {
        "District Name":   "district",
        "Block Name":      "block",
        "School Name":     "school_name",
        "UDISE Code":      "school_id",
        "School Category": "school_category",
    }
    for cls in CLASS_LIST:
        col_map[f"Class {cls}(Boys)"]  = f"boys_{cls}"
        col_map[f"Class {cls}(Girls)"] = f"girls_{cls}"
        col_map[f"Class {cls}(Total)"] = f"total_{cls}"

    all_rows = []
    for year, fpath in GOVT_FILES.items():
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found — skipping {year}")
            continue
        raw = pd.read_excel(fpath)
        # Strip stray quotes from column names (e.g. "'District Name'" -> "District Name")
        raw.columns = [c.strip().strip("'") for c in raw.columns]
        raw.rename(columns=col_map, inplace=True)
        raw["year"] = year

        # Compute total boys/girls/enrollment per row
        raw["total_boys"]       = sum(pd.to_numeric(raw.get(f"boys_{c}",  0), errors="coerce").fillna(0)
                                      for c in CLASS_LIST).astype(int)
        raw["total_girls"]      = sum(pd.to_numeric(raw.get(f"girls_{c}", 0), errors="coerce").fillna(0)
                                      for c in CLASS_LIST).astype(int)
        raw["total_enrollment"] = sum(pd.to_numeric(raw.get(f"total_{c}", 0), errors="coerce").fillna(0)
                                      for c in CLASS_LIST).astype(int)

        # Keep only standard columns that exist
        keep = [c for c in STD_COLS if c in raw.columns]
        all_rows.append(raw[keep])
        print(f"  {year}: {len(raw)} schools loaded from {fpath}")

    df = pd.concat(all_rows, ignore_index=True)

    # Ensure all STD_COLS present
    for col in STD_COLS:
        if col not in df.columns:
            df[col] = 0

    df = df[STD_COLS].copy()

    skip = {"school_id", "school_name", "district", "block", "year", "school_category"}
    for col in [c for c in STD_COLS if c not in skip]:
        df[col] = _safe_int(df[col])

    # Add school type columns before saving
    df = apply_school_types(df)
    _print_school_type_dist(df)

    os.makedirs("data", exist_ok=True)
    df.to_excel(save_path, index=False)
    print(f"\nMerged govt data saved: {save_path}")
    print(f"Total records: {len(df)} | Schools: {df['school_id'].nunique()} | Years: {sorted(df['year'].unique().tolist())}")
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_govt_data()
    report = validate_data(df)
    print(f"\n4-year analysis period: {report['years'][0]} to {report['years'][-1]}")
    print("Validation complete. Real govt data ready for analysis.")
