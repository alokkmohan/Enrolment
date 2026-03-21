DATA_FILE = "data/enrollment_govt.xlsx"   # real govt data (merged from 4 year files)

GOVT_FILES = {
    "2022-23": "OUTPUT/Enrollment_2022-23_Govt_Matched.xlsx",
    "2023-24": "OUTPUT/Enrollment_2023-24_Govt_Matched.xlsx",
    "2024-25": "OUTPUT/Enrollment_2024-25_Govt_Matched.xlsx",
    "2025-26": "OUTPUT/Enrollment_2025-26_Govt_Matched.xlsx",
}

YEARS = ["2022-23", "2023-24", "2024-25", "2025-26"]

YEAR_PAIRS = [
    ("2022-23", "2023-24"),
    ("2023-24", "2024-25"),
    ("2024-25", "2025-26"),
]

BASE_YEAR    = "2022-23"   # Sabse purana
CURRENT_YEAR = "2025-26"  # Sabse naya
PREV_YEAR    = "2024-25"  # Ek saal purana

CLASS_LIST = [8, 9, 10, 11, 12]

OUTPUT_PATHS = {
    "L1": "outputs/L1_STATE/",
    "L2": "outputs/L2_DISTRICT/",
    "L3": "outputs/L3_BLOCK/",
    "L4": "outputs/L4_SCHOOL/",
    "L5": "outputs/L5_MASTER/",
}

RISK_THRESHOLDS = {
    "low_enrollment"  : 100,
    "small_school"    : 50,
    "micro_school"    : 25,
    "transition_fail" : 70,
    "transition_weak" : 85,
    "dropout_high"    : 30,
    "cv_stable"       : 5,
    "cv_volatile"     : 15,
    "growth_high"     : 15,
    "growth_decline"  : -5,
    "girls_imbalance" : 0.8,
}
