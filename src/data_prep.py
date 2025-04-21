import pandas as pd

def load_and_merge(trip_path: str, person_path: str, household_path: str) -> pd.DataFrame:
    trip = pd.read_csv(trip_path)
    person = pd.read_csv(person_path)[['person_id', 'age', 'education']]
    household = pd.read_csv(household_path)[['hh_id', 'income_aggregate']]

    df = (
        trip
        .merge(person, on='person_id', how='left')
        .merge(household, on='hh_id', how='left')
    )

    columns = [
        'person_id', 'hh_id', 'age', 'education', 'income_aggregate',
        'depart_time', 'arrive_time', 'dwell_time_min',
        'o_purpose_category', 'd_purpose_category',
        'd_distance_home', 'd_distance_work', 'd_distance_school'
    ]
    return df[columns].copy()

def prepare_activity_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['depart_time'] = pd.to_datetime(df['depart_time'])
    df['arrive_time'] = pd.to_datetime(df['arrive_time'])
    df.sort_values(['person_id', 'depart_time'], inplace=True)

    df['end_time'] = (
        df
        .groupby('person_id')['depart_time']
        .shift(-1)
    )
    df['activity_num'] = (
        df
        .groupby('person_id')
        .cumcount()
        + 1
    )

    cols = [
        'person_id', 'hh_id', 'age', 'education', 'income_aggregate',
        'activity_num', 'arrive_time', 'end_time', 'd_purpose_category',
        'd_distance_home', 'd_distance_work', 'd_distance_school'
    ]
    activity_df = df[cols].copy()
    activity_df.rename(columns={
        'arrive_time': 'start_time',
        'd_purpose_category': 'activity_purpose'
    }, inplace=True)

    return activity_df

if __name__ == '__main__':
    trip_path = "BATS_2019_Trip.csv"
    person_path = "BATS_2019_Person.csv"
    household_path = "BATS_2019_Household.csv"

    df_merged = load_and_merge(trip_path, person_path, household_path)
    activity_df = prepare_activity_df(df_merged)

    print(activity_df.head())