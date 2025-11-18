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

# Drop rows where 'date' is missing. These rows are all missing lat/long data as well.
df = df.dropna(subset=['date'])

# Create day of the week column
df['day_of_week'] = df['date'].dt.day_name()

# Drop old columns
df.drop(columns=['year', 'month', 'day'], inplace=True)

# Extract hour and rename column
df['hour'] = df['hour_minute'].str.split(':').str[0].astype(int)

# Drop the original 'hour_minute' column
df = df.drop(columns=['hour_minute'])

# Move new columns to the front
cols = ['date', 'day_of_week', 'hour'] + [c for c in df.columns if c not in ['date', 'day_of_week', 'hour']]
df = df[cols]

print(f"Cleaned date columns")


# ------------------------------------------------------
# 7. DROP IRRELEVANT COLUMNS FROM THE TABLE
# ------------------------------------------------------

# Columns removed from each dataset
cols_to_drop = {
    "users dataset": [
        'vehicle_id', 'vehicle_number', 'seat_position', 'user_category', 'sex',
        'birth_year', 'trip_purpose', 'safety_device_1', 'safety_device_2',
        'safety_device_3', 'pedestrian_location', 'pedestrian_action',
        'pedestrian_alone', 'user_id'
    ],
    "places dataset": [
        'road_number', 'numerical_index_road', 'alphanumeric_index_road',
        'road_ref_1', 'road_ref_2', 'width_central_reservation',
        'width_carriageway'
    ],
    "details dataset": [
        'accident_uid', 'road_address', 'commune'
    ]
}

# Flatten all columns into one list
all_drop_cols = [col for cols in cols_to_drop.values() for col in cols]

# Drop them in one go (ignoring missing columns just in case)
df = df.drop(columns=all_drop_cols, errors='ignore')

print(f"Dropped unnecessary columns from users, places, and details datasets. "
      f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ------------------------------------------------------
# 8. RECATEGORIZING COLUMNS
# ------------------------------------------------------

# Recategorize road_category column

road_category_mapping = {
    1: 'Major Roads',
    2: 'Major Roads',
    3: 'Secondary Roads',
    7: 'Secondary Roads',
    4: 'Local & Access Roads',
    6: 'Local & Access Roads',
    5: 'Other / Off-Network',
    9: 'Other / Off-Network'
}

df['road_category'] = df['road_category'].map(road_category_mapping)


# Recategorize road layout column
# Unknown values impluted with 'Two Way' (most common value)

road_layout_mapping = {
    -1: 'Two Way',
    1: 'One Way',
    2: 'Two Way',
    3: 'Multi Lane',
    4: 'Multi Lane'
}

df['road_layout'] = df['road_layout'].map(road_layout_mapping)


# Convert num_lanes from object to int
def clean_to_int(x):
    try:
        # Remove whitespace, then convert to int
        return int(str(x).strip())
    except:
        # If conversion fails, classify as -1
        return -1

df['num_lanes'] = df['num_lanes'].apply(clean_to_int)

# Replace lanes that are 0 or -1 with 2 (most common value)
df.loc[(df['num_lanes'] < 1), 'num_lanes'] = 2


# Recategorize reserved_lane column
# Unknown values impluted with 'No value' (most common value)

reserved_lane_mapping = {
    -1: 'No value',
    0: 'No value',
    1: 'Cycle Lane',
    2: 'Cycle Lane',
    3: 'Reserved Lane'
}

df['reserved_lane'] = df['reserved_lane'].map(reserved_lane_mapping)


# Recategorize road_profile column
# Unknown values impluted with 'flat' (most common value)

road_profile_mapping = {
    -1: 'Flat',
    1: 'Flat',
    2: 'Slope / Near Slope',
    3: 'Slope / Near Slope',
    4: 'Slope / Near Slope'
}

df['road_profile'] = df['road_profile'].map(road_profile_mapping)


# Recategorize road_shape column
# Unknown values impluted with 'straight' (most common value)

road_shape_mapping = {
    -1: 'Straight',
    1: 'Straight',
    2: 'Curved',
    3: 'Curved',
    4: 'Curved'
}

df['road_shape'] = df['road_shape'].map(road_shape_mapping)


# Recategorize surface_condition column
# Unknown values impluted with 'normal' (most common value)

surface_condition_mapping = {
    -1: 'Normal',
    1: 'Normal',
    2: 'Wet / Slippery',
    3: 'Wet / Slippery',
    4: 'Wet / Slippery',
    5: 'Wet / Slippery',
    6: 'Wet / Slippery',
    7: 'Wet / Slippery',
    8: 'Wet / Slippery',
    9: 'Wet / Slippery'
}

df['surface_condition'] = df['surface_condition'].map(surface_condition_mapping)


# Recategorize infrastructure column
# Unknown values impluted with 'No value' (most common value)

infrastructure_mapping = {
    -1: 'No value',
    0: 'No value',
    1: 'Tunnel / Bridge',
    2: 'Tunnel / Bridge',
    3: 'Intersections',
    4: 'Intersections',
    5: 'Intersections',
    6: 'Intersections',
    7: 'Other',
    8: 'Other',
    9: 'Other'
}

df['infrastructure'] = df['infrastructure'].map(infrastructure_mapping)


# Recategorize road_location column
# Unknown values impluted with 'Road' (most common value)

road_location_mapping = {
    -1: 'Road',
    0: 'Road',
    1: 'Road',
    2: 'Reserved Lanes',
    3: 'Reserved Lanes',
    4: 'Cyclist / Pedestrian',
    5: 'Cyclist / Pedestrian',
    6: 'Reserved Lanes',
    8: 'Other'
}

df['road_location'] = df['road_location'].map(road_location_mapping)


# Cleaning the speed limit column
# Round 'speed_limit' to nearest 10
df['speed_limit'] = ((df['speed_limit'] / 10).round(0) * 10).astype(int)

# There are rows where speed limit is between 130 and 200. Impute it with 130, assuming these are highways.
df.loc[(df['speed_limit'] > 130) & (df['speed_limit'] < 200), 'speed_limit'] = 130

# There are rows where speed limit is over 200. Impute it with the median speed (50kmh), assuming these are input errors.
median_speed = df[df['speed_limit']<=130]['speed_limit'].median()

df.loc[(df['speed_limit'] > 130), 'speed_limit'] = median_speed

# Impute missing speed limits with 50.
df.loc[(df['speed_limit'] < 1), 'speed_limit'] = 50


# Recategorize light_conditions column

light_conditions_mapping = {
    1: 'Day',
    2: 'Twilight',
    3: 'Night',
    4: 'Night',
    5: 'Night'
}

df['light_conditions'] = df['light_conditions'].map(light_conditions_mapping)

# 6 rows with missing light conditions
def classify_light_condition(hour):
    if 7 <= hour <= 18:
        return "Day"
    elif 5 <= hour <= 6 or 19 <= hour <= 20:
        return "Twilight"
    else:
        return "Night"

mask = df['light_conditions'].isna()

df.loc[mask, 'light_conditions'] = (
    df.loc[mask, 'hour']
            .apply(classify_light_condition)
)


# Keep only rows with numeric department codes
# This removes rows corresponding to Corsica ('2A'/'2B') and overseas territories
df = df[df['department'].astype(str).str.isdigit().fillna(False)]

# Define Île-de-France department codes
idf_departments = {75, 77, 78, 91, 92, 93, 94, 95}

# Convert to integer
df['department'] = df['department'].astype(int)

# Keep only IDF rows
df = df[df['department'].isin(idf_departments)]


# Rename value of urban_area column
urban_area_mapping = {
    1: 'Outside urban area',
    2: 'Inside urban area'
}

df['urban_area'] = df['urban_area'].map(urban_area_mapping)


# Recategorize intersection_type column
# Unknown values impluted with 'No junction' (most common value)

intersection_type_mapping = {
    -1: 'No junction',
    1: 'No junction',
    2: 'Simple junction',
    3: 'Simple junction',
    4: 'Simple junction',
    5: 'Complex junction',
    6: 'Complex junction',
    7: 'Complex junction',
    8: 'Other junction',
    9: 'Other junction'
}

df['intersection_type'] = df['intersection_type'].map(intersection_type_mapping)


# Recategorize weather column based on visibiltiy and road condition
# Unknown values impluted with 'Normal' (most common value)

weather_mapping = {
    -1: 'Normal Visibility',
     1: 'Normal Visibility',
     8: 'Normal Visibility',
     2: 'Reduced Traction',
     3: 'Reduced Traction',
     4: 'Reduced Traction',
     5: 'Reduced Visibility',
     6: 'Normal Visibility',
     7: 'Normal Visibility',
     9: 'Reduced Visibility'
}

df['weather'] = df['weather'].map(weather_mapping)


# Recategorize collision_type column
# Unknown values impluted with '2-car collision' (most common value)

collision_type_mapping = {
    -1: '2-car collision',  # Not specified
     1: '2-car collision',  # Head-on
     2: '2-car collision',  # Rear-end
     3: '2-car collision',  # Side collision
     4: 'Multi-car collision', # Chain collision
     5: 'Multi-car collision', # Multiple collisions
     6: 'Multi-car collision', # Other collision
     7: 'No collision'      # No collision
}

df['collision_type'] = df['collision_type'].map(collision_type_mapping)


# Convert latitude and longitude from strings with comma decimal separator to float
df['latitude'] = df['latitude'].str.replace(',', '.').astype(float)
df['longitude'] = df['longitude'].str.replace(',', '.').astype(float)

print('Recategorized columns')


# ------------------------------------------------------
# 9. DROP DUPLICATE ROWS
# ------------------------------------------------------

# Remove fully duplicated rows to ensure each row is unique across all columns
# This will avoid data leakage
df = df.drop_duplicates(keep='first')

print(f"Dropped duplicate rows. "
      f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
