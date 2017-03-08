#!/usr/bin/env python3
# system modules

# internal modules

# external modules
import numpy as np
import pandas as pd

def convolve(x,filter,intuitively=True):
    """ wrapper around np.convolve. calculate convolution of x and filter.
    "valid" mode, only full overlap is used. filter may be flipped to bypass
    np.convolve strange default behaviour.
    Args:
        x (1d np.array): base series
        filter (1d np.array): moved filter
        intuitively (bool): Flip the filter to revert numpy.convolve's filter
        flip. Defaults to True.  
    Returns:
        res = np.array: convolution of x and filter
    """
    assert len(x.shape) <= 1, "x must be one-dimensional"
    assert len(filter.shape) <= 1, "filter must be one-dimensional"

    # convolve only with full overlap 
    # reverse filter (np.convolve reverses) if desired
    if intuitively:
        res = np.convolve( x, filter[::-1], mode = "valid" )
    else:
        res = np.convolve( x, filter, mode = "valid" )

    return res


def convolution_matrix_fixed_filter(length,filter):
    """ Create a matrix that a series of length 'length' may be multplied with
    to obtain the convolution of the series with 'filter'. ("valid" mode - only
    full overlap)
    Args:
        length (int): Length of the series to be convolved
        filter (1d np.array): moved filter
    Returns:
        res = 2d numpy.array (matrix)
    """
    assert isinstance(length, int), "length has to be int"
    assert len(filter.shape) <= 1, "filter must be one-dimensional"

    # start with empty matrix
    conv_length = max(filter.size, length) - min(filter.size, length) + 1
    mat = np.zeros([conv_length, length])

    # loop over rows
    # first row scalar product with series yields first element of convolution
    # second row scalar product with series yields second element of convolution
    # and so on...
    for row in range(mat.shape[0]):
        mat[row,range(row,row+filter.size)] = filter

    return mat


def convolution_matrix_continuous_kernel(kernel,lags):
    """ Create a convolution matrix based on a continuous kernel and lags.
    Args:
        kernel (callable): callable that takes lags as np.array 
            and returns np.array of same shape with kernel values
        lags (1d np.array): lags of convoluted series in seconds. Will be sorted
            and unified!
    Returns:
        mat = 2d numpy.array Matrix
    """
    assert hasattr(kernel, '__call__'), "kernel has to callable"
    assert lags.shape[0] == lags.size, "lags has to be 1d numpy.array"

    lags = np.unique(lags) # sort and unify the lags

    # start with empty square matrix
    mat = np.ma.masked_all((lags.size,lags.size)) # masked 
    mat.fill_value = 0 # fill with zeros later on

    # loop over rows
    for row in range(mat.shape[0]):
        # indices where the kernel is used for the convolution
        indices = list(range(row+1)) # range(0) -> [], so use row+1
        # get the lags of this index region
        lags_here = lags[list(reversed(indices))]
        # fill the matrix with the kernel values at these lags
        mat[row,indices] = kernel(lags_here)

    return mat.filled()


def convolve_series_with_continuous_kernel(series, kernel):
    """ Convolve base_series with a continuous kernel 
    Args:
        base_series (pd.Series with DatetimeIndex): The series to be convolved
        kernel (callable): callable that takes lags as np.array 
            and returns np.array of same shape with kernel values
    Returns:
        convolved = pd.Series with DatetimeIndex: Convolved series
    """
    # make sure arguments are sane
    assert hasattr(series,'index'), 'series has no index'
    assert hasattr(kernel, '__call__'), 'kernel is not callable'

    # get the lags
    lags = series.index.values.astype('float64') * 1e-9
    lags = lags - lags.min()
    # get the convolution matrix
    conv_matrix = convolution_matrix_continuous_kernel( 
        kernel = kernel, lags = lags)
    # series convolved
    convolved = np.dot( conv_matrix, series.values)
    # create series
    res = pd.Series( 
        data  = convolved,
        index = series.index.copy()
        )
    # return
    return(res)
