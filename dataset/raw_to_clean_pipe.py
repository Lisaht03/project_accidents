import os
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# ------------------------------------------------------
# 1. PROJECT ROOT
# ------------------------------------------------------

cwd = os.getcwd()
if os.path.basename(cwd) == "dataset":
    PROJECT_ROOT = os.path.dirname(cwd)
else:
    PROJECT_ROOT = cwd

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

print(f"Raw data directory: {RAW_DATA_DIR}")


# ------------------------------------------------------
# 2. Helper function to load all CSVs matching a prefix
# ------------------------------------------------------
def load_all_years(prefix):
    """
    Loads all CSVs where filename starts with prefix + '-' (e.g., caracteristiques-2020.csv).
    Returns a single concatenated DataFrame.
    """
    files = [
        f for f in os.listdir(RAW_DATA_DIR)
        if f.startswith(prefix + "-") and f.endswith(".csv")
    ]

    if not files:
        raise FileNotFoundError(f"No files found for prefix '{prefix}' in /data")

    dfs = []
    for file in sorted(files):   # sorted ensures year order
        path = os.path.join(RAW_DATA_DIR, file)
        print(f"  → Loading {file}")
        dfs.append(pd.read_csv(path, sep=";"))

    combined = pd.concat(dfs, ignore_index=True)
    print(f"{prefix}: Loaded {combined.shape[0]:,} rows")
    return combined


# ------------------------------------------------------
# 3. LOAD RAW DATA
# ------------------------------------------------------
details = load_all_years("caracteristiques")
places  = load_all_years("lieux")
users   = load_all_years("usagers")

print("\nAll raw files loaded successfully.\n")


# ------------------------------------------------------
# 4. ADD MAX INJURY SEVERITY AND PEOPLE INVOLVED TO USERS DF
# ------------------------------------------------------

# Count number of users involved per accident
users["users_involved"] = (
    users.groupby("Num_Acc")["Num_Acc"].transform("count")
)

# Select row with max 'grav' per accident
users = users.loc[
    users.groupby("Num_Acc")["grav"].idxmax()
]


# ------------------------------------------------------
# 5. MERGE ALL DATASETS
# ------------------------------------------------------

df = (
    places
    .merge(users, on="Num_Acc", how="left")
    .merge(details, on="Num_Acc", how="left")
)

print(f"Merged dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")


# ------------------------------------------------------
# 6. RENAME COLUMNS
# ------------------------------------------------------

df.rename(columns={
    'Num_Acc': 'accident_number',
    'jour': 'day',
    'mois': 'month',
    'an': 'year',
    'hrmn': 'hour_minute',
    'lum': 'light_conditions',
    'dep': 'department',
    'com': 'commune',
    'agg': 'urban_area',
    'int': 'intersection_type',
    'atm': 'weather',
    'col': 'collision_type',
    'adr': 'road_address',
    'lat': 'latitude',
    'long': 'longitude',
    'Accident_Id': 'accident_uid',
    'catr': 'road_category',
    'voie': 'road_number',
    'v1': 'numerical_index_road',
    'v2': 'alphanumeric_index_road',
    'circ': 'road_layout',
    'nbv': 'num_lanes',
    'vosp': 'reserved_lane',
    'prof': 'road_profile',
    'pr': 'road_ref_1',
    'pr1': 'road_ref_2',
    'plan': 'road_shape',
    'lartpc': 'width_central_reservation',
    'larrout': 'width_carriageway',
    'surf': 'surface_condition',
    'infra': 'infrastructure',
    'situ': 'road_location',
    'vma': 'speed_limit',
    'id_vehicule': 'vehicle_id',
    'num_veh': 'vehicle_number',
    'place': 'seat_position',
    'catu': 'user_category',
    'grav': 'injury_severity',
    'sexe': 'sex',
    'an_nais': 'birth_year',
    'trajet': 'trip_purpose',
    'secu1': 'safety_device_1',
    'secu2': 'safety_device_2',
    'secu3': 'safety_device_3',
    'locp': 'pedestrian_location',
    'actp': 'pedestrian_action',
    'etatp': 'pedestrian_alone',
    'id_usager': 'user_id'
}, inplace=True)

print(f"Renamed columns")


# ------------------------------------------------------
# 6. CLEAN DATE COLUMNS
# ------------------------------------------------------

# Combine day, month, year columns
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

# Create day of the week column
df['day_of_week'] = df['date'].dt.day_name()

# Drop old columns
df.drop(columns=['year', 'month', 'day'], inplace=True)

# Move new columns to the front
cols = ['date', 'day_of_week'] + [c for c in df.columns if c not in ['date', 'day_of_week']]
df = df[cols]

print(f"Cleaned date columns")
