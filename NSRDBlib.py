import pandas as pd
import os
import requests

api_key = 'hYEIjf5E7wn7xSMuKQxkCEQ8nZ0LwLisVLAyUKRV'
api_user = 'David+Maranto'
api_user_email = 'd.maranto1122@gmail.com'
api_years = '2012, 2013, 2014, 2015, 2016'
api_lon, api_lat = '-79.3832', '43.6532'        # Toronto
api_interval = '60'     # 1hr
api_utc = 'false'       # local time

#region Functions
def request_data():     # data will be sent via email
    url = 'http://developer.nrel.gov/api/solar/nsrdb_psm3_download.json?api_key={}'.format(api_key)
    payload = 'names={0}&interval={1}&utc={2}&full_name={3}&email={4}&wkt=POINT({5}%20{6})'.format(api_years, api_interval, api_utc, api_user, api_user_email, api_lon, api_lat)
    headers = {
        'content-type': "application/x-www-form-urlencoded",
        'cache-control': "no-cache"
    }
    response = requests.request('POST', url, data=payload, headers=headers)
    print(response.text)


def preprocess_data():     # process and combine yearly data to comp
    data = pd.DataFrame()
    for file in sorted(os.listdir('./data/raw/')):
        print(file)
        data_raw = pd.read_csv('./data/raw/{}'.format(file), header=2, index_col=0)
        data = pd.concat([data, data_raw], axis=0, sort=False)

    data.drop(columns=['Month', 'Day', 'Hour', 'Minute', 'Fill Flag'], inplace=True)

    # Column renaming
    data.rename({
        'Temperature':'temp',
        'Clearsky DHI':'dhi_clear',
        'Clearsky DNI':'dni_clear',
        'Clearsky GHI':'ghi_clear',
        'Dew Point':'dew_point',
        'DHI':'dhi',
        'DNI':'dni',
        'GHI':'ghi',
        'Relative Humidity':'humidity_rel',
        'Solar Zenith Angle':'zenith_angle',
        'Surface Albedo':'albedo_sur',
        'Pressure':'pressure',
        'Precipitable Water':'precipitation',
        'Wind Direction':'wind_dir',
        'Wind Speed':'wind_speed',
        'Cloud Type':'cloud_type'}, axis=1, inplace=True)

    # Nan handler
    data['cloud_type'].fillna(method='ffill', inplace=True)      # categorical forward fill
    data.interpolate(method='linear', axis=0, limit_area='inside', inplace=True)        # linear interpolation

    # categorical encoding
    data = pd.get_dummies(data, dummy_na=False, columns=['cloud_type'])     # cloud type num assigned may not correspond w/ original designation

    print(data.head())
    data.to_csv('./data/comp.csv')
    print('Success!')


def get_data(features):
    data = pd.read_csv('./data/comp.csv', header=0, index_col=0)
    for col in list(data.columns.values):
        if col not in features:
            data.drop(columns=[col], inplace=True)

    return data
#endregion

if __name__ == '__main__':
    preprocess_data()