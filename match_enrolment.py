import pandas as pd

# File paths
enrolment_file = 'OUTPUT/GOVT_ENROLLMENT_2022-23.csv'
school_master_file = 'School Master.csv'
output_file = 'OUTPUT/GOVT_ENROLLMENT_MATCHED.csv'

# Read files
enrolment_df = pd.read_csv(enrolment_file)
school_master_df = pd.read_csv(school_master_file, encoding='latin1')

# Ensure UDISE column exists and is consistent
enrolment_udise_col = 'UDISE Code'
school_master_udise_col = 'UDISE Code'

# Filter matched UDISE codes
merged_df = pd.merge(
	enrolment_df,
	school_master_df[[school_master_udise_col, 'School Management', 'School Category']],
	left_on=enrolment_udise_col,
	right_on=school_master_udise_col,
	how='inner'
)

# Filter for School Management
filtered_df = merged_df[merged_df['School Management'] == 'Department of Education (Government School)']

# Save matched file
required_columns = [
	'District Name', 'Block Name', 'School Name', 'UDISE Code', 'School Management', 'School Category',
	'Class 8(Boys)', 'Class 8(Girls)', 'Class 8(Trans)', 'Class 8(Total)',
	'Class 9(Boys)', 'Class 9(Girls)', 'Class 9(Trans)', 'Class 9(Total)',
	'Class 10(Boys)', 'Class 10(Girls)', 'Class 10(Trans)', 'Class 10(Total)',
	'Class 11(Boys)', 'Class 11(Girls)', 'Class 11(Trans)', 'Class 11(Total)',
	'Class 12(Boys)', 'Class 12(Girls)', 'Class 12(Trans)', 'Class 12(Total)'
]
filtered_df = filtered_df[required_columns]
filtered_df.to_csv(output_file, index=False)

print(f"Matched and filtered enrolment file saved to {output_file}")
