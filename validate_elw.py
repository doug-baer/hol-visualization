import argparse
import math
import sys
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# EXAMPLE USE: validate_elw.py --pods pods.yaml --schedule schedule.csv 

### Pods YAML Example
# pods:
#  - Name: HOL-2001
#    NumVMs: 11
#    NumCPU: 163
#    TbRAM: 0.997
#    TbDisk: 7.68
#    StartupTime: 29
#   - Name: HOL-2002
#     NumVMs: 11
#     NumCPU: 163
#     TbRAM: 0.997
#     TbDisk: 10.42
#     StartupTime: 43

### Schedule CSV Example
#Room,TimeSlot,PodName
#Room1,09:00-11:00,HOL-2001
#Room1,11:00-13:00,HOL-2002
#Room1,13:00-15:00,HOL-2001
#Room1,15:00-17:00,HOL-2002


# Room capacity/scale multipliers (number of seats per room)
ROOM_MULTIPLIERS = {"Room1": 32, "Room2": 40, "Room3": 40}


def load_data(yaml_path, csv_path):
    """Load and parse YAML and CSV files."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as yf:
            pods_data = yaml.safe_load(yf)
            pods_df = pd.DataFrame(pods_data["pods"]).set_index("Name")

        # Fill missing StartupTime with 0 minutes by default
        if "StartupTime" not in pods_df.columns:
            pods_df["StartupTime"] = 0
        else:
            pods_df["StartupTime"] = pods_df["StartupTime"].fillna(0)

        schedule_df = pd.read_csv(csv_path)

        return pods_df, schedule_df

    except FileNotFoundError as e:
        print(f"Error: Could not find file -> {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data files: {e}")
        sys.exit(1)


def parse_time_to_minutes(time_str):
    """Convert 'HH:MM' string to total minutes from midnight."""
    h, m = map(int, time_str.split(":"))
    return h * 60 + m


def process_environment_load(blocks_df, schedule_df):
    """Aggregate hourly metrics scaled by room multipliers."""
    # Extend tracking hours from 06:00 to 19:00 to accommodate early startup times (normal start is 09:00)
    hours = [f"{h:02d}:00" for h in range(6, 20)]
    dimensions = ["NumVMs", "NumCPU", "TbRAM", "TbDisk"]

    # Initialize hourly aggregate counts
    hourly_totals = {
        hour: {dim: 0.0 for dim in dimensions} for hour in hours[:-1]
    }

    # Process schedule using DEPLOYMENT_POOL for the vPod lookup and ROOM_SIZE for the multiplier
    # TODO: process the DEPLOYMENT_POOL -- if it ends with "-vmworld" ... remove that to get PodName (if no valid pool, skip it)
    for _, row in schedule_df.iterrows():
        # room = str(row["ROOM"]).strip()
        pool_name = str(row["DEPLOYMENT_POOL"]).strip()
        
        # Ensure the deployment pool ends with '-vmworld', otherwise skip
        if not pool_name.endswith("-vmworld"):
            continue
            
        # Strip '-vmworld' to get the matching block name (e.g., 'HOL-2637-vmworld' -> 'HOL-2637')
        block_name = pool_name.removesuffix("-vmworld")

        start_str = str(row["START"]).strip()
        end_str = str(row["END"]).strip()
        
        # Use the ROOM_SIZE column as the multiplier (number of seats)
        try:
            multiplier = float(row["ROOM_SIZE"])
        except ValueError:
            multiplier = 1.0  # Fallback if ROOM_SIZE is not a number

        if block_name not in blocks_df.index:
            print(
                f"Warning: Pod '{block_name}' in schedule not found in YAML block definitions."
            )
            continue

        block_metrics = blocks_df.loc[block_name]
        startup_minutes = int(block_metrics.get("StartupTime", 0))
        
        # Calculate active start time including startup offset
        scheduled_start_min = parse_time_to_minutes(start_str)
        scheduled_end_min = parse_time_to_minutes(end_str)
        actual_start_min = scheduled_start_min - startup_minutes

        # Convert back to hourly buckets (floor start hour to capture partial warm-up hours)
        start_h = math.floor(actual_start_min / 60)
        end_h = math.ceil(scheduled_end_min / 60)

        # Scale metrics for each active hour
        for h in range(start_h, end_h):
            h_key = f"{h:02d}:00"
            if h_key in hourly_totals:
                for dim in dimensions:
                    scaled_value = block_metrics[dim] * multiplier
                    hourly_totals[h_key][dim] += scaled_value

    load_df = pd.DataFrame.from_dict(hourly_totals, orient="index")
    load_df.index.name = "Hour"
    return load_df


def plot_load(load_df, selected_date):
    """Visualize hourly utilization in a 2x2 multi-panel bar plot."""
    dimensions = ["NumVMs", "NumCPU", "TbRAM", "TbDisk"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    for idx, dim in enumerate(dimensions):
        ax = axes[idx]
        ax.bar(
            load_df.index,
            load_df[dim],
            color=colors[idx],
            width=0.6,
            alpha=0.85,
        )
        ax.set_title(
            f"Aggregate Load: {dim}", fontsize=12, fontweight="bold"
        )
        ax.set_ylabel("Total Units")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis='x',labelrotation=45)
        # add some extra room so that the labels don't crowd the top of the plot
        ax.margins(y=0.1)

        # annotate non-zero values above each bar
        for x, y in zip(load_df.index, load_df[dim]):
            if y > 0:
                ax.text(
                    x,
                    y + (y * 0.015),
                    f"{y:g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.suptitle(
        f"Hourly Utilization ({selected_date})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig('schedule.png', dpi=150, bbox_inches='tight')
    print("Chart saved as 'schedule.png'")
    plt.show()

def get_user_date_selection(schedule_df):
    """Prompt the user to select a date from the available dates in the schedule."""
    if "DAY" not in schedule_df.columns:
        print("Error: 'DAY' column not found in the CSV schedule.")
        sys.exit(1)

    unique_dates = sorted(schedule_df["DAY"].dropna().unique())
    
    if not unique_dates:
        print("Error: No valid dates found in the 'DAY' column.")
        sys.exit(1)

    print("\nAvailable dates in the schedule:")
    for idx, date_val in enumerate(unique_dates, start=1):
        print(f"  {idx}. {date_val}")

    while True:
        try:
            choice = input("\nEnter the number of the date you want to analyze: ")
            choice_idx = int(choice)
            if 1 <= choice_idx <= len(unique_dates):
                return unique_dates[choice_idx - 1]
            else:
                print(f"Please enter a number between 1 and {len(unique_dates)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize hourly environment utilization for Workshops using HOL vPods."
    )
    parser.add_argument(
        "--pods",
        default="pods.yaml",
        help="Path to YAML file with pod definitions (default: pods.yaml)",
    )
    parser.add_argument(
        "--schedule",
        default="schedule.csv",
        help="Path to CSV file with room schedule (default: schedule.csv)",
    )
    parser.add_argument(
        "--day",
        help="Bypass prompt and analyze a specific DAY directly (e.g., --day 08/25/2025)",
    )

    args = parser.parse_args()

    pods_df, schedule_df = load_data(args.pods, args.schedule)

    # Determine which day to analyze
    if args.day:
        if args.day not in schedule_df["DAY"].values:
            print(f"Error: The date '{args.day}' does not exist in the CSV file.")
            sys.exit(1)
        selected_day = args.day
    else:
        selected_day = get_user_date_selection(schedule_df)

    # Filter schedule for the selected day
    filtered_schedule_df = schedule_df[schedule_df["DAY"] == selected_day]

    # Process and plot (removed the old hardcoded ROOM_MULTIPLIERS)
    load_df = process_environment_load(pods_df, filtered_schedule_df)

    print("--- Scaled Hourly Utilization ---")
    print(load_df)

    plot_load(load_df, selected_day)