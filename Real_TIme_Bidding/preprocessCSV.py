# Loading the dependancies.
from datetime import time
import os
import pandas as pd
from tqdm import tqdm
import time


# Step 0:Creates click column.
def create_click_col(df1, df2):
    click = df1['bid_id'].isin(df2['bid_id'])
    df1['click'] = click
    # Dropping the log_type and bid_id column as it is useless from now on.
    df1.drop(columns=['bid_id', 'log_type'], axis=1, inplace=True)
    # Dropping nan values of click.
    df1.dropna(subset=['click'], inplace=True)
    # Reordering columns.
    cols = ['click'] + [col for col in df1 if col != 'click']
    df1 = df1[cols]
    return df1

# Step 1:Creates date and hour columns.


def get_date(df):
    df['timestamp'] = df['timestamp'].astype(str)
    df['timestamp'] = df.apply(lambda x: x['timestamp'][:-3], axis=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S')
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.dayofweek
    df.drop(columns=['timestamp'], axis=1, inplace=True)
    cols = ['day', 'hour', 'click'] + \
        [col for col in df if col not in ['day', 'hour', 'click']]
    df = df[cols]
    return df

# Step 2:Will be done dynaically in get_train.

# Step 3:Creating the os and browser column.


class user_agent(object):
    def __init__(self, df):
        self.df = df
        self.df.rename(columns={'user_agent': 'browser'}, inplace=True)
        self.df['browser'] = df['browser'].astype(str)
        self.cols = None

    def map_browser(self, agent):
        browsers = ['edge', 'trident', 'chrome', 'firefox', 'safari', 'opera']
        for browser in browsers:
            if browser in agent.lower():
                return 'ie' if browser == 'trident' else browser
        return 'other'

    def map_os(self, agent):
        os_list = ['windows', 'linux', 'mac os x']
        for os in os_list:
            if os in agent.lower():
                return os
        return 'other'

    def reorder_cols(self, target_index):
        cols = [col for col in self.df]
        cols[target_index], cols[-1] = cols[-1], cols[target_index]
        self.df = self.df[cols]
        return self.df

    def create_cols(self):
        self.df['os'] = self.df['browser'].map(
            lambda x: self.map_os(x), na_action=None)
        self.df['os'] = self.df['os'].astype('category')
        self.df['browser'] = self.df['browser'].map(
            lambda x: self.map_browser(x), na_action=None)
        self.df['browser'] = self.df['browser'].astype('category')
        self.df = self.reorder_cols(4)
        return self.df

# Step 4:Conversion to Numeric Features.


def numerise_features(df, int_cols=['click','ad_slot_width', 'ad_slot_height', 'ad_slot_floor_price', 'bidding_price', 'paying_price']):
    for feature in int_cols:
        df[feature] = df[feature].astype('int32')
    return df

# Step 5: Conversion to Category features.


def categorize_features(df, category_features=['day', 'hour', 'browser','os','ad_slot_visibility', 'ad_slot', 'ad_exchange', 'region_id']):
    for feature in category_features:
        df[feature] = df[feature].astype('category')
    return df


# A function which implemnts all the above steps and returns the training dataframe.
def get_data(path_clk, path_imp, mode='Train'):
    cols_to_be_dropped = ['ipinyou_id', 'ip_address', 'city_id', 'domain', 'url',
                          'anonymous_url_id', 'ad_slot_id', 'creative_id', 'key_page_url', 'user_tags']
    clk_df = pd.read_csv(path_clk)
    chunks = pd.read_csv(path_imp, chunksize=100000)
    df = pd.DataFrame()
    print('Begin preprocessing...')
    for chunk in tqdm(chunks,desc = 'Chunks Progress Bar'):
        # Step 0
        chunk = create_click_col(chunk, clk_df)
        # Step 1
        chunk = get_date(chunk)
        # Step 2
        chunk.drop(columns=cols_to_be_dropped, axis=1, inplace=True)
        # Step 3
        col_creator = user_agent(chunk)
        chunk = col_creator.create_cols()
        # Step 4
        chunk = numerise_features(chunk)
        # Step 5
        chunk = categorize_features(chunk)
        if mode == 'Test':
            chunk.drop(columns=['click'], axis=1, inplace=True)
        # Concatenation.
        df = pd.concat([df, chunk], ignore_index=True)
    print('Ufff! Finally, I am done.')
    return df
