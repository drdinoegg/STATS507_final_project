import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

def filter_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Remove income/education codes indicating missing or prefer-not-to-answer."""
    df = df[df['income_aggregate'] < 99]
    df = df[df['education'] < 99]
    return df

def map_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw codes into grouped categories."""
    # income
    income_map = {
        1: 'low_income', 2: 'low_income',
        3: 'mid_income', 4: 'mid_income',
        5: 'high_income', 6: 'high_income'
    }
    df['income_group'] = df['income_aggregate'].map(income_map)
    # age
    age_map = {
        1: 'child', 2: 'child', 3: 'child',
        4: 'young', 5: 'young',
        6: 'adult', 7: 'adult', 8: 'adult',
        9: 'senior', 10: 'senior'
    }
    df['age_group'] = df['age'].map(age_map)
    # education
    edu_map = {
        1: 'highschool_or_less', 2: 'highschool_or_less',
        3: 'some_college', 4: 'some_college', 5: 'some_college',
        6: 'bachelor_plus', 7: 'bachelor_plus'
    }
    df['education_group'] = df['education'].map(edu_map)
    return df

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and relative-day features."""
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time']   = pd.to_datetime(df['end_time'])
    # duration in minutes
    df['duration_min'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
    # day/hour/minute
    df['day_of_week']    = df['start_time'].dt.dayofweek
    df['hour_of_day']    = df['start_time'].dt.hour
    df['minute_of_hour'] = df['start_time'].dt.minute
    # relative day per person
    first = df.groupby('person_id')['start_time'].transform('min')
    df['relative_day'] = (df['start_time'] - first).dt.days + 1
    return df

def encode_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encoding for time and day-of-week."""
    seconds = df['hour_of_day']*3600 + df['minute_of_hour']*60 + df['start_time'].dt.second
    df['time_sin'] = np.sin(2*np.pi * seconds/86400)
    df['time_cos'] = np.cos(2*np.pi * seconds/86400)
    df['dow_sin']  = np.sin(2*np.pi * df['day_of_week']/7)
    df['dow_cos']  = np.cos(2*np.pi * df['day_of_week']/7)
    return df

def clean_and_select(df: pd.DataFrame) -> pd.DataFrame:
    """Drop original datetime columns and any rows with invalid negatives or NaNs."""
    # drop start/end
    df = df.drop(columns=['start_time', 'end_time'])
    # no negative for these cols
    nonneg = ['duration_min', 'd_distance_home', 'd_distance_work', 'd_distance_school']
    df = df[(df[nonneg] >= 0).all(axis=1)]
    # drop rows with NaN in any required column
    df = df.dropna()
    return df

class PersonDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, target_cols, seq_len=50):
        self.seq_len = seq_len
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.target_cols = target_cols
        self.pids = df['person_id'].unique()
        self.labels = df.groupby('person_id')[target_cols].first()
        self.df = df

    def pad(self, arr, pad_value=0):
        if len(arr) >= self.seq_len:
            return arr[:self.seq_len]
        pad = np.full((self.seq_len - len(arr), arr.shape[1]), pad_value)
        return np.vstack([arr, pad])

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        sub = self.df[self.df['person_id']==pid].sort_values('activity_num')
        cats = sub[self.cat_cols].values
        cont = sub[self.cont_cols].values
        cats = self.pad(cats, pad_value=0)
        cont = self.pad(cont, pad_value=0.0)
        label = self.labels.loc[pid].values
        return (torch.LongTensor(cats), torch.FloatTensor(cont)), torch.LongTensor(label)

def prepare_dataloaders(df, cat_cols, cont_cols, target_cols, batch_size=32):
    """Encode labels, scale cont, split train/val, and return DataLoaders."""
    # fill cont NaN
    df[cont_cols] = df[cont_cols].fillna(df[cont_cols].median())
    # encode cats & targets
    for c in cat_cols + target_cols:
        le = LabelEncoder().fit(df[c].astype(int))
        df[c] = le.transform(df[c].astype(int))
    # scale cont
    scaler = StandardScaler().fit(df[cont_cols])
    df[cont_cols] = scaler.transform(df[cont_cols])
    # split
    train, val = train_test_split(df, test_size=0.2, random_state=42)
    ds_train = PersonDataset(train, cat_cols, cont_cols, target_cols)
    ds_val   = PersonDataset(val,   cat_cols, cont_cols, target_cols)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)
    return loader_train, loader_val

if __name__ == '__main__':
    # 1) load merged and activity df from data_prep
    from data_prep import prepare_activity_df, load_and_merge

    df_merged = load_and_merge(
        "BATS_2019_Trip.csv",
        "BATS_2019_Person.csv",
        "BATS_2019_Household.csv"
    )
    act_df = prepare_activity_df(df_merged)

    # 2) feature engineering
    df1 = filter_invalid(act_df)
    df2 = map_categories(df1)
    df3 = extract_time_features(df2)
    df4 = encode_cyclical(df3)
    df_clean = clean_and_select(df4)

    # 3) create loaders
    cat_cols = ['activity_num', 'day_of_week', 'hour_of_day', 'minute_of_hour', 'relative_day']
    cont_cols = ['d_distance_home', 'd_distance_work', 'd_distance_school',
                 'duration_min', 'time_sin', 'time_cos', 'dow_sin', 'dow_cos']
    target_cols = ['age_group', 'education_group', 'income_group']

    train_loader, val_loader = prepare_dataloaders(df_clean, cat_cols, cont_cols, target_cols)

    # quick sanity check
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")