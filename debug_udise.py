import pandas as pd

# Read files
enrolment_df = pd.read_csv('d:\\Enrollment\\OUTPUT\\GOVT_ENROLLMENT_2022-23.csv')
school_master_df = pd.read_csv('d:\\Enrollment\\School Master.csv', encoding='latin1')

print('=== Enrollment UDISE Code (first 10) ===')
print(enrolment_df['UDISE Code'].head(10))
print(f'Data type: {enrolment_df["UDISE Code"].dtype}')

print('\n=== School Master UDISE Code (first 10) ===')
print(school_master_df['UDISE Code'].head(10))
print(f'Data type: {school_master_df["UDISE Code"].dtype}')

# Check for matching
enrol_udise = set(enrolment_df['UDISE Code'].astype(str).str.strip())
master_udise = set(school_master_df['UDISE Code'].astype(str).str.strip())

print(f'\nEnrollment UDISE count: {len(enrol_udise)}')
print(f'School Master UDISE count: {len(master_udise)}')
print(f'Matched count: {len(enrol_udise & master_udise)}')

# Show some samples
print('\nSample enrollment UDISE:')
print(list(enrol_udise)[:5])
print('\nSample school master UDISE:')
print(list(master_udise)[:5])

# Check difference
only_in_enrol = enrol_udise - master_udise
only_in_master = master_udise - enrol_udise

print(f'\n\nUDISE only in Enrollment: {len(only_in_enrol)}')
print(f'UDISE only in School Master: {len(only_in_master)}')

if only_in_enrol:
    print('\nSample UDISE only in Enrollment:')
    print(list(only_in_enrol)[:5])

if only_in_master:
    print('\nSample UDISE only in School Master:')
    print(list(only_in_master)[:5])
