#!/usr/bin/env python3
# system modules

# internal modules

# external modules
import numpy as np
import pandas as pd


#################
### functions ###
#################
def empty_series():
    """ return an empty timeseries, i.e. a pandas.Series with no values but
    with an empty pandas.DatetimeIndex as index.
    """
    series = pd.Series(
        data = [], # empty data
        index = pd.DatetimeIndex( # datetime index
            data=[],dtype="datetime64[ns]",name='time')
        )
    return series


def common_lags(timestamps1, timestamps2):
    """ Compute the merged lags of two timestamp series in seconds
    Args:
        timestamps1, timestamps2 (pd.DatetimeIndex): the timestamps 
    Returns:
        lags = 1d numpy.array: the merged lags
    """
    # get proper numpy.arrays of seconds of timestamps1 and timestamps2
    seconds1 = timestamps1.values.astype('float64') * 1e-9
    seconds2 = timestamps2.values.astype('float64') * 1e-9
    assert timestamps1.size, "no data in timestamps1"
    assert timestamps2.size, "no data in timestamps2"
    lags = np.concatenate((seconds1, seconds2)) # concatenate
    lags = np.unique(lags) # drop repetitions + sort
    lags = lags - lags.min() # only differences
    return lags

