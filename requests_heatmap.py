import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns

# Load the CSV file
df = pd.read_csv('./data/events_test.csv', sep=';')

# Parse timestamp fields (ISO format)
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

# Get date range
min_date = df['Start'].min().normalize()
max_date = df['End'].max().normalize() + timedelta(days=1)
print(f"Analysis period: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

# Create daily activity dataframe
date_range = pd.date_range(start=min_date, end=max_date, freq='D')
daily_activity = []

for date in date_range:
    next_date = date + timedelta(days=1)

    # Count events active during this day
    active_events = df[(df['Start'] < next_date) & (df['End'] >= date)]

    daily_activity.append({
        'Date': date,
        'Active_Events': len(active_events),
        'Total_Seats': active_events['Seats'].sum() if len(active_events) > 0 else 0,
        'Has_Event': 1 if len(active_events) > 0 else 0
    })

activity_df = pd.DataFrame(daily_activity)

# =====================================================
# OPTION 1: Daily Event Density Heatmap (Recommended)
# =====================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Schedule Gap Analysis - Training Events July-Oct 2026',
             fontsize=14, fontweight='bold')

# Top panel: Event count per day
sns.heatmap(activity_df[['Has_Event']].transpose(),
            cmap='Blues', ax=axes[0], cbar=False, annot=True, fmt='d')
axes[0].set_xticks([])
axes[0].set_yticks([0])
axes[0].set_yticklabels(['Event Present (1) / No Event (0)'])
axes[0].set_title('Daily Event Presence', fontweight='bold')

# Bottom panel: Number of concurrent events per day
sns.heatmap(activity_df[['Active_Events']].transpose(),
            cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Concurrent Events'},
            annot=True, fmt='d', vmin=0)
axes[1].set_xticks(range(0, len(date_range), 7))  # Mark every 7 days
axes[1].set_xticklabels([d.strftime('%b %d') for d in date_range[::7]], rotation=45)
axes[1].set_yticks([0])
axes[1].set_yticklabels(['Concurrent Events'])
axes[1].set_title('Number of Concurrent Events Per Day', fontweight='bold')

plt.tight_layout()
#plt.savefig('schedule_gap_analysis.png', dpi=150, bbox_inches='tight')
#print("\nGap analysis chart saved as 'schedule_gap_analysis.png'")
plt.show()

# =====================================================
# OPTION 2: Weekly/Monthly Aggregation View
# =====================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Group by week
activity_df['Week'] = activity_df['Date'].dt.isocalendar().week
weekly = activity_df.groupby('Week').agg({
    'Active_Events': ['mean', 'sum'],
    'Has_Event': 'sum',
    'Total_Seats': 'sum'
}).round(1)

# Simplify column names
weekly.columns = ['Avg_Daily_Events', 'Total_Events', 'Days_With_Activity', 'Total_Seats']
weekly.index.name = 'Week Number'

# Bar chart for weekly totals
weeks = weekly.index.values
weekly['Total_Events'].plot(kind='bar', ax=axes[0], color='#2196F3', alpha=0.8)
axes[0].set_xlabel('Week Number')
axes[0].set_ylabel('Total Events Scheduled')
axes[0].set_title('Weekly Event Count', fontweight='bold')
axes[0].tick_params(axis='x', rotation=0)
axes[0].grid(axis='y', alpha=0.3)

# Heatmap for monthly breakdown
activity_df['Month'] = activity_df['Date'].dt.month
#activity_df['Week'] = activity_df['Date'].dt.isocalendar().week
monthly_weekly = activity_df.pivot_table(
    values='Active_Events',
    index=activity_df['Date'].dt.day_name(),
    columns=['Month', 'Week'],
    aggfunc='mean', fill_value=0
)

# Monthly view
## TODO: Fix the aggregation here since the results seem too high

#monthly_totals = activity_df.groupby(activity_df['Date'].dt.month)['Active_Events'].sum()
monthly_totals = activity_df.groupby('Month')['Active_Events'].sum()
months = ['Jul', 'Aug', 'Sep', 'Oct']
month_nums = [7, 8, 9, 10]
monthly_data = [monthly_totals[m] if m in monthly_totals.index else 0 for m in month_nums]

axes[1].bar(months, monthly_data, color=['#FFCDD2', '#FFF9C4', '#C8E6C9', '#BBDEFB'])
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Total Active Events (Sum)')
axes[1].set_title('Monthly Event Volume', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
#plt.savefig('schedule_weekly_monthly.png', dpi=150, bbox_inches='tight')
#print("Weekly/monthly chart saved as 'schedule_weekly_monthly.png'")
plt.show()

# =====================================================
# GAP IDENTIFICATION REPORT
# =====================================================
print("\n" + "=" * 60)
print("SCHEDULE GAP ANALYSIS REPORT")
print("=" * 60)

# Find all gap periods (consecutive days with no events)
gap_periods = []
gap_start = None
gap_count = 0

for idx, row in activity_df.iterrows():
    if row['Has_Event'] == 0:
        if gap_start is None:
            gap_start = row['Date']
            gap_count = 1
        else:
            gap_count += 1
    else:
        if gap_count >= 1:  # Only record meaningful gaps
            gap_end = row['Date'] - timedelta(days=1)
            gap_periods.append({
                'Gap_Start': gap_start,
                'Gap_End': gap_end,
                'Gap_Days': gap_count
            })
            gap_start = None
            gap_count = 0

# Handle trailing gap
if gap_count >= 1:
    gap_end = activity_df['Date'].iloc[-1]
    gap_periods.append({
        'Gap_Start': gap_start,
        'Gap_End': gap_end,
        'Gap_Days': gap_count
    })

# Print gap details
print("\nIDENTIFIED GAPS (No Events Scheduled):")
print("-" * 60)

for i, gap in enumerate(gap_periods, 1):
    print(f"\nGap #{i}:")
    print(f"  Start: {gap['Gap_Start'].strftime('%Y-%m-%d')}")
    print(f"  End:   {gap['Gap_End'].strftime('%Y-%m-%d')}")
    print(f"  Duration: {gap['Gap_Days']} day(s)")

    # Show surrounding context
    prev_day = gap['Gap_Start'] - timedelta(days=1)
    next_day = gap['Gap_End'] + timedelta(days=1)

    prev_events = df[(df['End'] > prev_day) & (df['Start'] <= prev_day)].shape[0]
    next_events = df[(df['Start'] >= next_day) & (df['End'] >= next_day)].shape[0]

    print(f"  Previous day events: {prev_events}")
    print(f"  Following day events: {next_events}")

# Summary statistics
total_days = len(activity_df)
days_with_events = activity_df['Has_Event'].sum()
days_without_events = total_days - days_with_events

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(f"\nTotal Analysis Period: {total_days} days")
print(f"Days WITH Events:      {days_with_events} ({days_with_events / total_days * 100:.1f}%)")
print(f"Days WITHOUT Events:   {days_without_events} ({days_without_events / total_days * 100:.1f}%)")
print(f"Number of Gap Periods: {len(gap_periods)}")
print(f"Largest Gap:           {max([g['Gap_Days'] for g in gap_periods]) if gap_periods else 0} days")

# Events per month
print("\n" + "-" * 60)
print("EVENTS BY MONTH:")
print("-" * 60)
for month_num, month_name in [(7, 'July'), (8, 'August'), (9, 'September'), (10, 'October')]:
    month_events = df[(df['Start'].dt.month == month_num)].shape[0]
    month_days = sum(activity_df['Has_Event'][activity_df['Date'].dt.month == month_num])
    print(f"{month_name:12}: {month_events:3} events | {month_days:3} active days")

# Type breakdown
print("\n" + "-" * 60)
print("EVENTS BY TYPE:")
print("-" * 60)
type_counts = df['Type'].value_counts()
for event_type, count in type_counts.items():
    seats = df[df['Type'] == event_type]['Seats'].sum()
    print(f"{event_type:12}: {count:3} events | {seats:4} total seats")

print("\n" + "=" * 60)
print("REPORT COMPLETE")
print("=" * 60)

# =====================================================
# OPTIONAL: Export gap data to CSV for further analysis
# =====================================================
#gap_report_df = pd.DataFrame(gap_periods)
#gap_report_df['Gap_Start'] = gap_report_df['Gap_Start'].dt.strftime('%Y-%m-%d')
#gap_report_df['Gap_End'] = gap_report_df['Gap_End'].dt.strftime('%Y-%m-%d')
#gap_report_df.to_csv('schedule_gaps_report.csv', index=False)
#print("\nGap report exported to 'schedule_gaps_report.csv'")