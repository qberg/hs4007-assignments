#Python code of the feature extraction performed in notebook 2.

#Dependancies loading...
import pandas as pd
from pandas import get_dummies
import numpy as np



def make_floorprice_bins(df, nbins=10):
    '''
    A method that makes bins out of floor price values.
    '''
    bins = pd.DataFrame()
    labels = np.arange(1, nbins+1, 1)

    #Cutting up the floor price into bins.
    bins['floorprice_bin'] = pd.cut(df['ad_slot_floor_price'].values,
                                    bins=nbins,
                                    labels=labels)

    #Dropping the old feature.
    df = df.drop(columns=['ad_slot_floor_price'], axis=1)
    #Adding the new feature.
    df['floorprice_bin'] = bins['floorprice_bin'].values
    bins.pop('floorprice_bin')
    return df

def indicatorise_features(df,category_features=['day','hour', 'browser', 'os', 'ad_exchange',
                              'ad_slot_visibility', 'ad_slot','region_id','floorprice_bin']):
    dummies = {}
    for feature in category_features:
        dummies[feature] = get_dummies(df[feature], prefix=feature)
    df = df.join(list(dummies.values()))
    df.drop(columns=category_features, axis=1, inplace=True)
    return df

def featurise_data(data):
    '''
    Featurises the data passed as a dataframe.
    '''
    #Floor price feature.
    data = make_floorprice_bins(data)
    #Indicatrise the variables.
    data = indicatorise_features(data)
    return data