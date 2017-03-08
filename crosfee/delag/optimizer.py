#!/usr/bin/env python3
# system modules
import logging

# internal modules
from . import convolve
from . import kernels
from . import utils

# external modules
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.optimize




class KernelOptimizer(object):
    """ Class to optimize a LagKernel to a lagged timeseries based on an
    unlagged timeseries.  
    """
    def __init__(self, original = None, lagged = None, kernel = None):
        """ class constructor
        Args:
            original (pandas.Series): original (unlagged) timeseries
            lagged (pandas.Series): lagged timeseries
        """
        if original is None:
            original = utils.empty_series()
        if lagged is None:
            lagged = utils.empty_series()
        self.original = original
        self.lagged = lagged

        self.kernel = kernel

    ##################
    ### Properties ###
    ##################
    @property
    def logger(self):
        """ the logging.Logger used for logging.
        Defaults to logging.getLogger(__name__).
        """
        try: # try to return the internal property
            return self._logger 
        except AttributeError: # didn't work
            return logging.getLogger(__name__) # return default logger
    
    @logger.setter
    def logger(self, logger):
        assert isinstance(logger, logging.Logger), \
            "logger property has to be a logging.Logger"
        self._logger = logger

    @property
    def original(self):
        """ Original timeseries. pandas.Series with pandas.DatetimeIndex.
        """
        try:
            self._original
        except AttributeError:
            self._original = utils.empty_series() 
        return self._original

    @original.setter
    def original(self, series):
        assert isinstance(series, pd.Series) , \
            "original must be of type pandas.Series"
        assert isinstance(series.index, pd.DatetimeIndex) , \
            "original series index must be pandas.DateTimeIndex"
        # set attribute
        self._original = series
    
    @property
    def lagged(self):
        """ Lagged timeseries. pandas.Series with pandas.DatetimeIndex.
        """
        try:
            self._lagged
        except AttributeError:
            self._lagged = utils.empty_series()
        return self._lagged

    @lagged.setter
    def lagged(self, series):
        assert isinstance(series, pd.Series) , \
            "lagged must be of type pandas.Series"
        assert isinstance(series.index, pd.DatetimeIndex) , \
            "lagged series index must be pandas.DateTimeIndex"
        # set attribute
        self._lagged = series

    @property
    def kernel(self):
        """ The lag Kernel. Object of class LagKernel.
        """
        try:
            return self._kernel
        except AttributeError:
            self._kernel = kernels.LagKernel( # TODO What is the default?
                num_parameters = 1, parameters = [0]
                )
        return self._kernel

    @kernel.setter
    def kernel(self, newkernel):
        assert issubclass(newkernel.__class__, kernels.LagKernel) , \
            "lagged must be object of LagKernel or derivates"
        self._kernel = newkernel

    @property
    def common_lags(self):
        """ Compute the merged lags of original and lagged in seconds
        Returns:
            lags = 1d numpy.array: the merged lags
        """
        return common_lags(self.original.index, self.lagged.index)

    @property
    def original_lags(self):
        """ Return the original series' lags
        Returns:
            lags = 1d numpy.array: the original series' lags
        """
        lags = self.original.index.values.astype('float64') * 1e-9
        lags = lags - lags.min()
        return lags

    @property
    def convolution_matrix(self):
        """ Create a convolution matrix based on the current LagKernel and the
            timestamps of lagged and original timeseries.
        Returns:
            mat = 2d numpy.array Matrix
        """
        return convolve.convolution_matrix_continuous_kernel(
            lags = self.original_lags,
            kernel = self.kernel
            )
    @property
    def original_convolved(self):
        """ Convolve the original series with the current kernel
        Returns:
            series = pandas.Series with same index as original series
        """
        res = pd.Series(
            data = np.dot( self.convolution_matrix, self.original ),
            index = self.original.index,
            )
        return res

    @property
    def original_convolved_at_lagged(self):
        """ Convolve the original series with the current kernel and interpolate
            to the lagged times.
        Returns:
            series = pandas.Series with same index as lagged series
        """
        # interpolate original convolved series
        interpolator = scipy.interpolate.interp1d( 
            x = self.original.index.values.astype('float64') * 1e-9,
            y = self.original_convolved,
            kind = "linear",
            )
        # interpolate
        res = pd.Series( 
            data = interpolator( 
                self.lagged.index.values.astype('float64') * 1e-9 ),
            index = self.lagged.index,
            )
        return res

    @property
    def residual(self):
        """ Calculate the RMSE of the residuals of the convolution of the
        original series with the kernel and the lagged series.
        Returns:
            residual = np.array of size 1
        """
        # calculate residual
        rmse = np.sqrt( np.mean( 
            (self.lagged.values - self.original_convolved_at_lagged) ** 2 ) )

        return rmse
        
    ###############
    ### Methods ###
    ###############
    def optimize(self, method = None):
        """ Optimize the kernel's parameters based on the convolution of the
            original series with the kernel and then minimizing the residual
            to the lagged series.
        Args:
            method (Optional[str or None]): optimization method. 
                See scipy.optimize.minimize doc.
        """
        # just a wrapper around the residual property
        def lossfunc(parameters): 
            self.logger.debug("current parameters: {}".format(parameters))
            self.kernel.parameters = list(parameters) # set parameters
            res = self.residual # calculate residual
            self.logger.debug("rmse for parameters {}: {}".format(parameters,
                res))
            return res # return residual

        scipy.optimize.minimize( 
            lossfunc, # loss function
            self.kernel.parameters, # first guess
            method = method, # optimization method
            bounds = self.kernel.parameter_bounds, # parameter bounds
            )
    

    def __repr__(self):
        """ python representation of this object
        """
        # self.logger.debug("__repr__ called")
        reprstring = ("{classname}(\n" 
              "original = {original},\n"
              "lagged = {lagged},\n"
              "kernel = {kernel},\n"
              ")").format(
            classname="{module}.{name}".format(
                name=self.__class__.__name__,module=self.__class__.__module__),
            # original series
            original = ("{cls}(\n"
                "    data = {data},\n"
                "    index = {index}\n"
                "    )"
                ).format(cls=self.original.__class__.__name__,
                    data=np.array(self.original).__repr__(),
                    index=self.original.index.__repr__()
                    ),
            # lagged series
            lagged = ("{cls}(\n"
                "    data = {data},\n"
                "    index = {index}\n"
                "    )"
                ).format(cls=self.lagged.__class__.__name__,
                data=np.array(self.lagged).__repr__(),
                index=self.lagged.index.__repr__()
                ),
            kernel = self.kernel.__repr__(),
            )
        return reprstring
