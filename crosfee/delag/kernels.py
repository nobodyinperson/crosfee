#!/usr/bin/env python3
# system modules
import logging

# external modules
import numpy as np

class LagKernel(object):
    """ Base class for lag kernels
    """
    def __init__(self, num_parameters, parameters, max_lag = 0.0):
        """ Class constructor
        Args:
            num_parameters (Int): number of parameters
            parameters (list of Float): parameters
            max_lag Optional[Float]: maximum lag of the kernel. Defaults to
                zero which means infinite maximum lag.
        """
        self.num_parameters = num_parameters
        self.parameters = parameters
        self.max_lag = max_lag

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
    def max_lag(self):
        """ The number of parameters for this Kernel
        """
        try:
            return self._max_lag
        except AttributeError:
            self._max_lag = 0 # TODO sensible default?
        return self._max_lag

    @max_lag.setter
    def max_lag(self, newlag):
        try:    newlag = float(newlag)
        except: raise ValueError("max_lag property has to be numeric")
        assert newlag >= 0, \
            "max_lag property has to be positive"
        self._max_lag = newlag

    @property
    def num_parameters(self):
        """ The number of parameters for this Kernel
        """
        try:
            return self._num_parameters
        except AttributeError:
            self._num_parameters = 1 # TODO sensible default?
        return self._num_parameters

    @num_parameters.setter
    def num_parameters(self, newnum):
        assert isinstance(newnum, int), \
            "num_parameters property has to be Integer"
        self._num_parameters = newnum

    @property
    def parameters(self):
        """ The parameters for this Kernel
        """
        try:
            return self._parameters
        except AttributeError:
            self._parameters = self.default_parameters
        return self._parameters

    @parameters.setter
    def parameters(self, newparameters):
        assert isinstance(newparameters, list), \
            "parameters property has to be list of numbers"
        assert len(newparameters) == self.num_parameters, \
            ("parameters have wrong length: " 
            "expected {} but got {} parameters").format(
                self.num_parameters, len(newparameters))
        assert self.check_parameters(newparameters), \
            ("invalid parameters (You should actually not see this. Make sure " 
             "your LagKernel subclass implements check_parameters method that " 
             "returns True on success.)")
        self._parameters = newparameters

    @property
    def default_parameters(self):
        """ Default parameters. Subclasses should override this.
        """
        return [0] * self.num_parameters

    @property
    def parameter_bounds(self):
        """ Parameter bounds. Subclasses should override this.
        Returns:
            list of (min,max)-tuples for each parameter
        """
        return [(-np.Inf,np.Inf)] * self.num_parameters

    def check_parameters(self, parameters):
        """ Check if parameters are allowed. Subclasses may override this to
            implement more sophisticated parameter checks.
        """
        for param,bounds in zip(parameters,self.parameter_bounds):
            lower, upper = bounds
            assert lower <= param and param <= upper, \
                ("parameter[{}] = {} is out of bounds {}").format(
                    parameters.index(param),param,bounds)
        return True

    def value(self, lag):
        """ return the kernel value at that time lag. Lags higher
            than the max_lag property (if max_lag is not equal to zero), result
            in the value zero.
        Args:
            lag (np.array or masked_array): the lag(s) in seconds
        Returns:
            res = np.ma.masked_array: the kernel values, masked where lag was
                higher than max_lag. Use res.filled() to fill it with
                res.fill_value.
        """
        raise NotImplementedError("value method needs to be " 
            "overriden by subclasses!")

    def __call__(self, lag):
        """ when called, return the kernel value at that time lag. Lags higher
            than the max_lag property (if max_lag is not equal to zero), result
            in the value zero.
        Args:
            lag (np.array or masked_array): the lag(s) in seconds
        Returns:
            res = np.array: the kernel values, zero where lag was
                higher than max_lag. 
        """
        # make sure lag is a PROPER np.array masked outside the max_lag region
        # convert to np.array
        lag = np.array(lag)
        # handle stupid case: np.array(1).shape == () instead of (1,)
        if not lag.shape: lag.shape = (lag.size,)
        # bad values
        mask = np.logical_or(
            np.logical_and(
                lag > self.max_lag, 
                self.max_lag != 0), 
            lag < 0)
        # mask the lag outside the max_lag region
        lag = np.ma.masked_where(a = lag, condition = mask )
        lag.fill_value = 0 # masked values may be filled with zeros

        # calculate
        res = self.value(lag)

        res = np.ma.masked_where(a = res, 
            condition = np.logical_or(mask,res < 0))
        res.fill_value = 0 # masked values may be filled with zeros

        # return filled array
        if hasattr(res, 'filled'):
            return res.filled()
        else:
            return res

    # TODO: make shown arguments dependant on __init__ arguments.
    #       If a subclass doesn't need num_parameters as argument, don't show it
    def __repr__(self):
        """ python representation of this object
        """
        reprstring = ("{classname}(\n" 
              "    num_parameters = {num_parameters},\n"
              "    max_lag = {max_lag},\n"
              "    parameters = {parameters},\n"
              "    )").format(
            classname="{module}.{name}".format(
                name=self.__class__.__name__,module=self.__class__.__module__),
            # number of parameters
            num_parameters = self.num_parameters.__repr__(),
            # parameters
            parameters = self.parameters.__repr__(),
            # max_lag
            max_lag = self.max_lag.__repr__(),
            )
        return reprstring


class LagKernelConstant(LagKernel):
    """ Constant kernel
    """
    def __init__(self,parameters = [1], max_lag = 0):
        """ Class constructor
        Args:
            parameters Optional[list of float]: the constant [1].
            max_lag Optional[Float]: maximum lag of the kernel. Defaults to
                zero which means infinite maximum lag.
        """
        super().__init__(
            num_parameters = 1, 
            parameters = parameters, 
            max_lag = max_lag)

    @property
    def default_parameters(self):
        """ Default parameters. Subclasses should override this.
        """
        return [1]

    @property
    def parameter_bounds(self):
        """ Only positive intercept, only negative slope
        Returns:
            list of (min,max)-tuples for each parameter
        """
        return [(0,np.Inf)]

    def value(self, lag):
        """ Linearly decreasing, zero where it would be below zero
        """
        res = np.empty_like(lag) # masked array like lags
        res[:] = self.parameters[0] # fill with parameter value
        return res # filled array



class LagKernelLinear(LagKernel):
    """ Linear-type kernel
    """
    def __init__(self,parameters = [1,-1], max_lag = 0):
        """ Class constructor
        Args:
            parameters Optional[list of float]: the parameters 
                [intercept, slope].  Defaults to [1,-1].
            max_lag Optional[Float]: maximum lag of the kernel. Defaults to
                zero which means infinite maximum lag.
        """
        super().__init__(
            num_parameters = 2, 
            parameters = parameters, 
            max_lag = max_lag)

    @property
    def default_parameters(self):
        """ Default parameters. Subclasses should override this.
        """
        return [1,-1]

    @property
    def parameter_bounds(self):
        """ Only positive intercept, only negative slope
        Returns:
            list of (min,max)-tuples for each parameter
        """
        return [(0,np.Inf),(-np.Inf,0)]

    def value(self, lag):
        """ Linearly decreasing, zero where it would be below zero
        """
        intercept, slope = self.parameters # parameters
        res = intercept + slope * lag # linear
        res = np.ma.masked_where(a = res, condition = res < 0) # mask below zero
        res.fill_value = 0 # fill masked values with zero
        return res.filled() # fill with zeros


class LagKernelWeibull(LagKernel):
    """ Weibull-distribution-like lag kernel
    """
    def __init__(self,parameters = [1,1], max_lag = 0):
        """ Class constructor
        Args:
            parameters Optional[list of float]: the parameters [scale, shape].
                Defaults to [1,1].
            max_lag Optional[Float]: maximum lag of the kernel. Defaults to
                zero which means infinite maximum lag.
        """
        super().__init__(
            num_parameters = 2, 
            parameters = parameters, 
            max_lag = max_lag)

    @property
    def default_parameters(self):
        """ Default parameters. Subclasses should override this.
        """
        return [1,1]

    @property
    def parameter_bounds(self):
        """ Only positive parameters are allowed.
        Returns:
            list of (min,max)-tuples for each parameter
        """
        return [(0,np.Inf)] * self.num_parameters

    def value(self, lag):
        """ Weibull distribution formula taken from
            https://en.wikipedia.org/wiki/Weibull_distribution
        """
        # calculate
        scale, shape = self.parameters
        res = ( shape / scale ) * ( lag / scale ) ** ( shape - 1 ) \
            * np.exp( - ( lag / scale ) ** shape ) 
        return res

class LagKernelPolynomGeneral(LagKernel):
    """ Polynomial lag kernel
    """
    def __init__(self,parameters = [1,-1], max_lag = 0):
        """ Class constructor
        Args:
            parameters Optional[list of float]: the parameters [scale, shape].
                Defaults to [1,1].
            max_lag Optional[Float]: maximum lag of the kernel. Defaults to
                zero which means infinite maximum lag.
        """
        super().__init__(
            num_parameters = len(parameters), 
            parameters = parameters, 
            max_lag = max_lag)

    @property
    def default_parameters(self):
        """ Default parameters. Subclasses should override this.
        """
        return [1,1]

    @property
    def parameter_bounds(self):
        """ Only positive parameters are allowed.
        Returns:
            list of (min,max)-tuples for each parameter
        """
        return [(-np.Inf,np.Inf)] * self.num_parameters

    def value(self, lag):
        """ Polynom
        """
        # calculate
        res = 0
        for i in range(len(self.parameters)):
            res = res + self.parameters[i] * lag ** (i)
        return res
