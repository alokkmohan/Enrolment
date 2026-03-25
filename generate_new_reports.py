"""
Generate all new report CSV files from dashboard_data.json
Saves into outputs/L1_STATE, L2_DISTRICT, L3_BLOCK, L4_SCHOOL, L5_MASTER
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import json, csv, os, statistics
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, 'dashboard', 'dashboard_data.json'), 'r', encoding='utf-8') as f:
    D = json.load(f)

schools      = D.get('all_schools', [])
dist_data    = D.get('district_data', [])
block_data   = D.get('block_health', [])
trend        = D.get('enrollment_trend', [])
kpi          = D.get('state_kpis', {})

def sf(v):
    try: return float(v or 0)
    except: return 0.0

def pct(a, b):
    return round((a - b) / b * 100, 1) if b else 0

def write_csv(rel_path, rows, fields=None):
    if not rows:
        print(f"  [SKIP] {rel_path} — no data")
        return
    path = os.path.join(BASE, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"  [OK] {rel_path} ({len(rows)} rows)")

# ─── Build district year-wise aggregates from schools ────────────────────────
dist_yr = defaultdict(lambda: dict(e22=0, e23=0, e24=0, e25=0, cnt=0))
for s in schools:
    d = (s.get('district') or s.get('District') or '').strip()
    if not d: continue
    dist_yr[d]['e22'] += sf(s.get('Enroll_2223'))
    dist_yr[d]['e23'] += sf(s.get('Enroll_2324'))
    dist_yr[d]['e24'] += sf(s.get('Enroll_2425'))
    dist_yr[d]['e25'] += sf(s.get('Enroll_2526'))
    dist_yr[d]['cnt'] += 1

dist_info = {(d.get('District') or '').strip(): d for d in dist_data}

def dist_row(name, v):
    e22, e23, e24, e25 = v['e22'], v['e23'], v['e24'], v['e25']
    info = dist_info.get(name, {})
    return {
        'District': name,
        'Schools': v['cnt'],
        'Enroll_2022_23': int(e22), 'Enroll_2023_24': int(e23),
        'Enroll_2024_25': int(e24), 'Enroll_2025_26': int(e25),
        'YoY_Change': int(e25 - e24),
        'YoY_Change_pct': pct(e25, e24),
        'Growth_4yr_pct': pct(e25, e22),
        'Boys_Growth_pct': info.get('Boys_Growth_pct', ''),
        'Girls_Growth_pct': info.get('Girls_Growth_pct', ''),
        'District_Rank': info.get('District_Rank', ''),
    }

all_dist = [dist_row(d, v) for d, v in dist_yr.items() if d]

# ─── Build block year-wise aggregates from schools ───────────────────────────
blk_yr = defaultdict(lambda: dict(dist='', e22=0, e23=0, e24=0, e25=0,
                                   boys22=0, girls22=0, boys25=0, girls25=0, cnt=0))
for s in schools:
    b = (s.get('block') or s.get('Block') or '').strip()
    d = (s.get('district') or s.get('District') or '').strip()
    if not b: continue
    e22, e23, e24, e25 = sf(s.get('Enroll_2223')), sf(s.get('Enroll_2324')), sf(s.get('Enroll_2425')), sf(s.get('Enroll_2526'))
    gr = sf(s.get('Girls_Ratio_2526') or 50)
    blk_yr[b]['dist'] = d
    blk_yr[b]['e22'] += e22; blk_yr[b]['e23'] += e23
    blk_yr[b]['e24'] += e24; blk_yr[b]['e25'] += e25
    blk_yr[b]['girls25'] += round(e25 * gr / 100)
    blk_yr[b]['boys25']  += e25 - round(e25 * gr / 100)
    blk_yr[b]['girls22'] += round(e22 * gr / 100)
    blk_yr[b]['boys22']  += e22 - round(e22 * gr / 100)
    blk_yr[b]['cnt'] += 1

blk_info = {(b.get('Block') or '').strip(): b for b in block_data}

def blk_row(name, v):
    e22, e23, e24, e25 = v['e22'], v['e23'], v['e24'], v['e25']
    info = blk_info.get(name, {})
    return {
        'District': v['dist'], 'Block': name, 'Schools': v['cnt'],
        'Enroll_2022_23': int(e22), 'Enroll_2023_24': int(e23),
        'Enroll_2024_25': int(e24), 'Enroll_2025_26': int(e25),
        'YoY_Change': int(e25 - e24),
        'YoY_Change_pct': pct(e25, e24),
        'Growth_4yr_pct': pct(e25, e22),
        'Zone': info.get('Block_Zone', ''),
        'Avg_Growth_pct': info.get('Avg_Growth_pct', ''),
    }

all_blk = [blk_row(b, v) for b, v in blk_yr.items() if b]

SCH_FIELDS = ['school_id', 'school_name', 'district', 'block', 'School_Type',
              'Class_Range_Label', 'Enroll_2223', 'Enroll_2324', 'Enroll_2425',
              'Enroll_2526', 'Overall_Growth_pct', 'Girls_Ratio_2526',
              'Risk_Level', 'Trend_Tag', 'Alert_Count']

def sch_row(s):
    return {f: s.get(f, '') for f in SCH_FIELDS}

# ════════════════════════════════════════════════════════════
# LEVEL 1 — STATE
# ════════════════════════════════════════════════════════════
print("\n── L1: STATE ──")

# Net YoY change
yoy_rows = []
for i, t in enumerate(trend):
    prev = trend[i-1]['total'] if i > 0 else None
    chg  = t['total'] - prev if prev else 0
    yoy_rows.append({
        'Year': t['year'], 'Enrollment': t['total'],
        'YoY_Change': chg if i else 0,
        'YoY_Change_pct': pct(t['total'], prev) if prev else 0,
        'Status': 'Base Year' if not i else ('Growth' if chg >= 0 else 'Decline')
    })
write_csv('outputs/L1_STATE/state_yoy_change.csv', yoy_rows)

# CAGR
if len(trend) >= 2:
    base, last, n = trend[0]['total'], trend[-1]['total'], len(trend) - 1
    cagr = round(((last / base) ** (1 / n) - 1) * 100, 2) if base else 0
    write_csv('outputs/L1_STATE/state_cagr.csv', [{
        'Metric': 'CAGR_4yr_pct', 'Value_pct': cagr,
        'Base_Year': trend[0]['year'], 'Base_Enrollment': base,
        'Latest_Year': trend[-1]['year'], 'Latest_Enrollment': last,
        'Interpretation': f"Enrollment growing at {cagr}% per year on average"
    }])

# Boys vs Girls trend (approximate using Girls_Ratio from schools)
yr_gender = {yr: dict(boys=0, girls=0, total=0) for yr in ['2022-23','2023-24','2024-25','2025-26']}
col_map = {'2022-23':'Enroll_2223','2023-24':'Enroll_2324','2024-25':'Enroll_2425','2025-26':'Enroll_2526'}
for s in schools:
    gr = sf(s.get('Girls_Ratio_2526') or 50)
    for yr, col in col_map.items():
        e = sf(s.get(col))
        g = round(e * gr / 100)
        yr_gender[yr]['girls'] += g
        yr_gender[yr]['boys']  += e - g
        yr_gender[yr]['total'] += e

gender_trend_rows = []
for yr in ['2022-23','2023-24','2024-25','2025-26']:
    v = yr_gender[yr]
    t = v['total']
    gender_trend_rows.append({
        'Year': yr, 'Boys': int(v['boys']), 'Girls': int(v['girls']),
        'Total': int(t),
        'Girls_Ratio_pct': round(v['girls'] / t * 100, 1) if t else 0,
        'Gender_Gap': int(v['boys'] - v['girls'])
    })
write_csv('outputs/L1_STATE/state_gender_trend.csv', gender_trend_rows)

# District decline count by year
decline_summary = []
for yr_lbl, cur_col, prv_col in [
    ('2023-24','e23','e22'), ('2024-25','e24','e23'), ('2025-26','e25','e24')
]:
    cnt_decl  = sum(1 for v in dist_yr.values() if v[cur_col] < v[prv_col] and v[prv_col] > 0)
    cnt_grow  = sum(1 for v in dist_yr.values() if v[cur_col] > v[prv_col])
    cnt_same  = len(dist_yr) - cnt_decl - cnt_grow
    decline_summary.append({
        'Year': yr_lbl, 'Total_Districts': len(dist_yr),
        'Districts_Growing': cnt_grow, 'Districts_Declining': cnt_decl, 'Districts_Stable': cnt_same
    })
write_csv('outputs/L1_STATE/state_district_decline_summary.csv', decline_summary)

# ════════════════════════════════════════════════════════════
# LEVEL 2 — DISTRICT
# ════════════════════════════════════════════════════════════
print("\n── L2: DISTRICT ──")

# Increased YoY
write_csv('outputs/L2_DISTRICT/dist_increased_yoy.csv',
    sorted([r for r in all_dist if r['YoY_Change_pct'] > 0], key=lambda x: -x['YoY_Change_pct']))

# Top 10 growing (4yr)
write_csv('outputs/L2_DISTRICT/dist_top10_growing.csv',
    sorted(all_dist, key=lambda x: -x['Growth_4yr_pct'])[:10])

# Bottom 10 declining (4yr)
write_csv('outputs/L2_DISTRICT/dist_bottom10_declining.csv',
    sorted(all_dist, key=lambda x: x['Growth_4yr_pct'])[:10])

# 2-yr continuous growth: e24>e23 AND e25>e24
write_csv('outputs/L2_DISTRICT/dist_2yr_continuous_growth.csv',
    sorted([r for r in all_dist if dist_yr[r['District']]['e24'] > dist_yr[r['District']]['e23']
            and dist_yr[r['District']]['e25'] > dist_yr[r['District']]['e24']],
           key=lambda x: -x['Growth_4yr_pct']))

# 3-yr continuous growth: e23>e22 AND e24>e23 AND e25>e24
write_csv('outputs/L2_DISTRICT/dist_3yr_continuous_growth.csv',
    sorted([r for r in all_dist if dist_yr[r['District']]['e23'] > dist_yr[r['District']]['e22']
            and dist_yr[r['District']]['e24'] > dist_yr[r['District']]['e23']
            and dist_yr[r['District']]['e25'] > dist_yr[r['District']]['e24']],
           key=lambda x: -x['Growth_4yr_pct']))

# 2-yr continuous decline: e24<e23 AND e25<e24
write_csv('outputs/L2_DISTRICT/dist_2yr_continuous_decline.csv',
    sorted([r for r in all_dist if dist_yr[r['District']]['e24'] < dist_yr[r['District']]['e23']
            and dist_yr[r['District']]['e25'] < dist_yr[r['District']]['e24']],
           key=lambda x: x['YoY_Change_pct']))

# 3-yr continuous decline
write_csv('outputs/L2_DISTRICT/dist_3yr_continuous_decline.csv',
    sorted([r for r in all_dist if dist_yr[r['District']]['e23'] < dist_yr[r['District']]['e22']
            and dist_yr[r['District']]['e24'] < dist_yr[r['District']]['e23']
            and dist_yr[r['District']]['e25'] < dist_yr[r['District']]['e24']],
           key=lambda x: x['YoY_Change_pct']))

# Both boys AND girls declined
write_csv('outputs/L2_DISTRICT/dist_both_declined.csv',
    sorted([r for r in all_dist if sf(r['Boys_Growth_pct']) < 0 and sf(r['Girls_Growth_pct']) < 0],
           key=lambda x: x['Growth_4yr_pct']))

# Girls improved, Boys declined
write_csv('outputs/L2_DISTRICT/dist_girls_up_boys_down.csv',
    sorted([r for r in all_dist if sf(r['Boys_Growth_pct']) < 0 and sf(r['Girls_Growth_pct']) > 0],
           key=lambda x: x['Boys_Growth_pct']))

# District recovery: was declining (e24<e23) but now growing (e25>e24)
write_csv('outputs/L2_DISTRICT/dist_recovery.csv',
    sorted([r for r in all_dist if dist_yr[r['District']]['e24'] < dist_yr[r['District']]['e23']
            and dist_yr[r['District']]['e25'] > dist_yr[r['District']]['e24']],
           key=lambda x: -x['YoY_Change_pct']))

# Growth volatility (std-dev of 3 YoY rates)
vol_rows = []
for d, v in dist_yr.items():
    if not d: continue
    rates = []
    for cur, prv in [('e23','e22'),('e24','e23'),('e25','e24')]:
        if v[prv] > 0: rates.append(pct(v[cur], v[prv]))
    if len(rates) >= 2:
        vol_rows.append({
            'District': d, 'Volatility_Score': round(statistics.stdev(rates), 2),
            'Enroll_2025_26': int(v['e25']), 'Growth_4yr_pct': pct(v['e25'], v['e22']),
            'YoY_Rates': ' | '.join(f"{r:+.1f}%" for r in rates)
        })
write_csv('outputs/L2_DISTRICT/dist_growth_volatility.csv',
    sorted(vol_rows, key=lambda x: -x['Volatility_Score']))

# ════════════════════════════════════════════════════════════
# LEVEL 3 — BLOCK
# ════════════════════════════════════════════════════════════
print("\n── L3: BLOCK ──")

# Blocks increased YoY
write_csv('outputs/L3_BLOCK/blk_increased_yoy.csv',
    sorted([r for r in all_blk if r['YoY_Change_pct'] > 0], key=lambda x: -x['YoY_Change_pct']))

# 2-yr continuous decline
write_csv('outputs/L3_BLOCK/blk_2yr_continuous_decline.csv',
    sorted([r for r in all_blk if blk_yr[r['Block']]['e24'] < blk_yr[r['Block']]['e23']
            and blk_yr[r['Block']]['e25'] < blk_yr[r['Block']]['e24']],
           key=lambda x: x['YoY_Change_pct']))

# 3-yr continuous decline
write_csv('outputs/L3_BLOCK/blk_3yr_continuous_decline.csv',
    sorted([r for r in all_blk if blk_yr[r['Block']]['e23'] < blk_yr[r['Block']]['e22']
            and blk_yr[r['Block']]['e24'] < blk_yr[r['Block']]['e23']
            and blk_yr[r['Block']]['e25'] < blk_yr[r['Block']]['e24']],
           key=lambda x: x['YoY_Change_pct']))

# Sharp decline >10%
write_csv('outputs/L3_BLOCK/blk_sharp_decline_10pct.csv',
    sorted([r for r in all_blk if r['YoY_Change_pct'] < -10], key=lambda x: x['YoY_Change_pct']))

# Top performing blocks (top 30 by 4yr growth)
write_csv('outputs/L3_BLOCK/blk_top_performing.csv',
    sorted(all_blk, key=lambda x: -x['Growth_4yr_pct'])[:30])

# Consistent growth (3 years)
write_csv('outputs/L3_BLOCK/blk_consistent_growth.csv',
    sorted([r for r in all_blk if blk_yr[r['Block']]['e23'] > blk_yr[r['Block']]['e22']
            and blk_yr[r['Block']]['e24'] > blk_yr[r['Block']]['e23']
            and blk_yr[r['Block']]['e25'] > blk_yr[r['Block']]['e24']],
           key=lambda x: -x['Growth_4yr_pct']))

# Boys / Girls declined (block-level approximation using school gender ratios)
boys_decl_blk, girls_decl_blk = [], []
for b, v in blk_yr.items():
    if not b: continue
    bg = pct(v['boys25'], v['boys22'])
    gg = pct(v['girls25'], v['girls22'])
    row = {
        'District': v['dist'], 'Block': b,
        'Boys_2022_23': int(v['boys22']), 'Boys_2025_26': int(v['boys25']), 'Boys_Growth_pct': bg,
        'Girls_2022_23': int(v['girls22']), 'Girls_2025_26': int(v['girls25']), 'Girls_Growth_pct': gg,
        'Total_2025_26': int(v['e25']), 'Zone': blk_info.get(b, {}).get('Block_Zone', '')
    }
    if bg < 0: boys_decl_blk.append(row)
    if gg < 0: girls_decl_blk.append(row)

write_csv('outputs/L3_BLOCK/blk_boys_declined.csv', sorted(boys_decl_blk, key=lambda x: x['Boys_Growth_pct']))
write_csv('outputs/L3_BLOCK/blk_girls_declined.csv', sorted(girls_decl_blk, key=lambda x: x['Girls_Growth_pct']))

# Orange zone blocks
write_csv('outputs/L3_BLOCK/blk_orange_zone.csv',
    sorted([r for r in all_blk if r['Zone'] == 'ORANGE'], key=lambda x: x['YoY_Change_pct']))

# ════════════════════════════════════════════════════════════
# LEVEL 4 — SCHOOL
# ════════════════════════════════════════════════════════════
print("\n── L4: SCHOOL ──")

# Schools increased YoY
sch_inc = sorted([s for s in schools if sf(s.get('Enroll_2425')) > 0
                  and sf(s.get('Enroll_2526')) > sf(s.get('Enroll_2425'))],
                 key=lambda x: -(sf(x.get('Enroll_2526')) - sf(x.get('Enroll_2425'))))
write_csv('outputs/L4_SCHOOL/sch_increased_yoy.csv', [sch_row(s) for s in sch_inc])

# 2-yr continuous decline
sch_2yr = sorted([s for s in schools
                  if sf(s.get('Enroll_2324')) > 0 and sf(s.get('Enroll_2425')) < sf(s.get('Enroll_2324'))
                  and sf(s.get('Enroll_2425')) > 0 and sf(s.get('Enroll_2526')) < sf(s.get('Enroll_2425'))],
                 key=lambda x: sf(x.get('Enroll_2526')))
write_csv('outputs/L4_SCHOOL/sch_2yr_continuous_decline.csv', [sch_row(s) for s in sch_2yr])

# Sharp decline >20% YoY
sch_sharp = sorted([s for s in schools
                    if sf(s.get('Enroll_2425')) > 0
                    and pct(sf(s.get('Enroll_2526')), sf(s.get('Enroll_2425'))) < -20],
                   key=lambda x: pct(sf(x.get('Enroll_2526')), sf(x.get('Enroll_2425'))))
write_csv('outputs/L4_SCHOOL/sch_sharp_decline_20pct.csv', [sch_row(s) for s in sch_sharp])

# Top growth schools (all growing, sorted)
sch_top = sorted([s for s in schools if sf(s.get('Overall_Growth_pct', 0)) > 0],
                 key=lambda x: -sf(x.get('Overall_Growth_pct', 0)))
write_csv('outputs/L4_SCHOOL/sch_top_growth.csv', [sch_row(s) for s in sch_top])

# Both declined: overall 4yr decline AND also 2025-26 < 2024-25
sch_both = sorted([s for s in schools
                   if sf(s.get('Overall_Growth_pct', 0)) < 0
                   and sf(s.get('Enroll_2425')) > 0
                   and sf(s.get('Enroll_2526')) < sf(s.get('Enroll_2425'))],
                  key=lambda x: sf(x.get('Overall_Growth_pct', 0)))
write_csv('outputs/L4_SCHOOL/sch_both_declined.csv', [sch_row(s) for s in sch_both])

# Low enrollment <50 students (2025-26)
sch_low = sorted([s for s in schools
                  if 0 < sf(s.get('Enroll_2526')) < 50],
                 key=lambda x: sf(x.get('Enroll_2526')))
write_csv('outputs/L4_SCHOOL/sch_low_enrollment_50.csv', [sch_row(s) for s in sch_low])

# ════════════════════════════════════════════════════════════
# LEVEL 5 — CROSS-CUTTING (L5_MASTER)
# ════════════════════════════════════════════════════════════
print("\n── L5: CROSS-CUTTING ──")

# Sudden spike or drop (>30% YoY change in any year)
sudden = []
for s in schools:
    e22, e23, e24, e25 = sf(s.get('Enroll_2223')), sf(s.get('Enroll_2324')), sf(s.get('Enroll_2425')), sf(s.get('Enroll_2526'))
    flags = []
    for yr, cur, prv in [('2023-24', e23, e22), ('2024-25', e24, e23), ('2025-26', e25, e24)]:
        if prv > 0 and abs(pct(cur, prv)) > 30:
            flags.append(f"{yr}: {pct(cur,prv):+.1f}%")
    if flags:
        row = sch_row(s)
        row['Spike_Drop_Details'] = ' | '.join(flags)
        row['Flag_Count'] = len(flags)
        sudden.append(row)
sudden.sort(key=lambda x: -x['Flag_Count'])
write_csv('outputs/L5_MASTER/sudden_spike_drop_schools.csv', sudden)

# Missing data (any year = 0 but other years have data)
missing = []
for s in schools:
    vals = [sf(s.get(c)) for c in ['Enroll_2223','Enroll_2324','Enroll_2425','Enroll_2526']]
    if any(v == 0 for v in vals) and any(v > 0 for v in vals):
        row = sch_row(s)
        row['Missing_Years'] = ', '.join(
            yr for yr, v in zip(['2022-23','2023-24','2024-25','2025-26'], vals) if v == 0)
        missing.append(row)
write_csv('outputs/L5_MASTER/missing_data_schools.csv', missing)

# Gender gap schools: Girls ratio > 65% or < 35%
gender_gap = sorted([s for s in schools
                     if sf(s.get('Enroll_2526', 0)) > 0
                     and (sf(s.get('Girls_Ratio_2526', 50)) > 65 or sf(s.get('Girls_Ratio_2526', 50)) < 35)],
                    key=lambda x: abs(sf(x.get('Girls_Ratio_2526', 50)) - 50), reverse=True)
write_csv('outputs/L5_MASTER/gender_gap_schools.csv', [sch_row(s) for s in gender_gap])

print("\nAll reports generated successfully!")
