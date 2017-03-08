#!/usr/bin/env python3
# system modules
import unittest
import logging

# import authentication module
from crosfee import delag

# import test data
from .test_data import *
from .test_flow import *

# external modules
import numpy as np
import scipy.interpolate
import pandas as pd

# skip everything
SKIPALL = False # by default, don't skip everything

################################
### test the LagKernel class ###
################################
class LagKernelTest(BasicTest):
    pass

class LagKernelWeibullTest(LagKernelTest):
    def setUp(self):
        # a lagkernel
        self.lagkernel = delag.kernels.LagKernelWeibull()

    @testname("value test")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_values(self):
        for scale,shape,lag,res in zip(
            WEIBULL_TEST_DATA.get("scale"),
            WEIBULL_TEST_DATA.get("shape"),
            WEIBULL_TEST_DATA.get("lag"),
            WEIBULL_TEST_DATA.get("res"),
            ):
            self.lagkernel.parameters = [scale,shape]
            # value should equal expected value
            self.assertEqual(self.lagkernel(lag), res)

class LagKernelConstantTest(LagKernelTest):
    def setUp(self):
        # a lagkernel
        self.lagkernel = delag.kernels.LagKernelConstant()

    @testname("value test")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_values(self):
        for constant,lag,res,max_lag in zip(
            CONSTANT_TEST_DATA.get("constant"),
            CONSTANT_TEST_DATA.get("lag"),
            CONSTANT_TEST_DATA.get("res"),
            CONSTANT_TEST_DATA.get("max_lag"),
            ):
            # set parameters
            self.lagkernel.parameters = [constant]
            self.lagkernel.max_lag = max_lag
            # value should equal expected value
            self.assertEqual(self.lagkernel(lag), res)


class LagKernelLinearTest(LagKernelTest):
    def setUp(self):
        # a lagkernel
        self.lagkernel = delag.kernels.LagKernelLinear()

    @testname("value test")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_values(self):
        for intercept,slope,lag,res in zip(
            LINEAR_TEST_DATA.get("intercept"),
            LINEAR_TEST_DATA.get("slope"),
            LINEAR_TEST_DATA.get("lag"),
            LINEAR_TEST_DATA.get("res"),
            ):
            self.lagkernel.parameters = [intercept,slope]
            # value should equal expected value
            self.assertEqual(self.lagkernel(lag), res)

#########################
### utility functions ###
#########################
class CommonLagsTest(BasicTest):
    @testname("common lags")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_common_lags_all(self):
        n = 10
        series1 = pd.DatetimeIndex(list(range(n)),dtype='datetime64[ns]')
        series2 = pd.DatetimeIndex(list(range(n,2*n)),dtype='datetime64[ns]')
        # compute common lags
        lags = delag.utils.common_lags(series1, series2)
        expected = pd.DatetimeIndex(list(range(2*n)),dtype='datetime64[ns]')
        expected = (expected.values.astype('float64') \
            - expected.values.astype('float64').min()) * 1e-9
        self.assertTrue(np.allclose(lags,expected))

    @testname("common lags partly overlap")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_common_lags_overlap(self):
        n = 10
        offset = 5
        series1 = pd.DatetimeIndex(list(range(offset,n+offset)),
            dtype='datetime64[ns]')
        series2 = pd.DatetimeIndex(list(range(n,2*n)),dtype='datetime64[ns]')
        # compute common lags
        lags = delag.utils.common_lags(series1, series2)
        expected = pd.DatetimeIndex(list(range(offset,2*n)),
            dtype='datetime64[ns]')
        expected = (expected.values.astype('float64') \
            - expected.values.astype('float64').min()) * 1e-9
        self.assertTrue(np.allclose(lags,expected))

    @testname("common lags full overlap")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_common_lags_full_overlap(self):
        n = 10
        series1 = pd.DatetimeIndex(list(range(n)),dtype='datetime64[ns]')
        series2 = series1.copy() # exact same series
        # compute common lags
        lags = delag.utils.common_lags(series1, series2)
        expected = series1.copy() # result should be same
        expected = (expected.values.astype('float64') \
            - expected.values.astype('float64').min()) * 1e-9
        self.assertTrue(np.allclose(lags,expected))

#########################
### convolution tests ###        
#########################
class ConvolutionWrapperTest(BasicTest):
    @testname("convolve wrapper function")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_convolve(self):
        for x,filter,res in zip(
            CONVOLUTION_TEST_DATA.get("x"),
            CONVOLUTION_TEST_DATA.get("filter"),
            CONVOLUTION_TEST_DATA.get("res"),
            ):
            conv = delag.convolve.convolve(x=x,filter=filter)
            # value should equal expected value
            self.assertTrue(np.allclose(conv, res))
    
class ConvolutionMatrixFixedFilterTest(BasicTest):
    @testname("convolution matrix creation function")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_creation(self):
        for length,filter,res in zip(
            CONVOLUTION_MATRIX_TEST_DATA.get("length"),
            CONVOLUTION_MATRIX_TEST_DATA.get("filter"),
            CONVOLUTION_MATRIX_TEST_DATA.get("res"),
            ):
            # create matrix
            matrix = delag.convolve.convolution_matrix_fixed_filter(
                length=length, filter=filter)
            # value should equal expected value
            self.assertTrue(np.allclose(matrix, res))

    @testname("convolution via matrix with fixed filter")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_convolve(self):
        for length,filter,res in zip(
            CONVOLUTION_MATRIX_TEST_DATA.get("length"),
            CONVOLUTION_MATRIX_TEST_DATA.get("filter"),
            CONVOLUTION_MATRIX_TEST_DATA.get("convres"),
            ):
            # create matrix
            matrix = delag.convolve.convolution_matrix_fixed_filter(
                length=length, filter=filter)
            # create series
            x = np.array(range(length))
            # convolve via matrix
            conv = np.dot(matrix,x)
            # value should equal expected value
            self.assertTrue(np.allclose(conv, res))

class ConvolutionMatrixFromContinuousKernelTest(BasicTest):
    @classmethod
    def kernel_one(cls,lags): 
        return np.ones_like(lags) # return ones in same shape

    @testname("convolution of simple series via matrix from continuous " 
        "constant kernel")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_convolve_constant_kernel_range_series(self):
        series = np.array(range(10))
        lags = np.array(range(0,series.size)) # create lags
        # calculate the cumsum
        cumsum = np.cumsum(series)
        # calculate the convolution
        mat = delag.convolve.convolution_matrix_continuous_kernel(
            kernel = self.kernel_one, lags = lags)
        conv = np.dot( mat, series)
        self.assertTrue(np.allclose(conv,cumsum)) # should be equal

    @testname("convolution of random series via matrix from continuous " 
        "constant kernel")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_convolve_constant_kernel_random_series(self):
        series = np.random.random(10)
        lags = np.array(range(0,series.size)) # create lags
        # calculate the cumsum
        cumsum = np.cumsum(series)
        # calculate the convolution
        mat = delag.convolve.convolution_matrix_continuous_kernel(
            kernel = self.kernel_one, lags = lags)
        conv = np.dot( mat, series)
        self.assertTrue(np.allclose(conv,cumsum)) # should be equal

    @testname("constant kernel")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_common_lags_all(self):
        n = 10
        base_range = list(range(n))
        base_series   = pd.Series(
            data  = np.array(base_range),
            index = pd.DatetimeIndex(base_range,dtype='datetime64[ns]'))

        self.logger.debug("base_series: \n{}".format(base_series))

        # convolve base_series with kernel 
        # timestamps
        res = delag.convolve.convolve_series_with_continuous_kernel( 
            series = base_series,
            kernel = self.kernel_one)
        self.logger.debug("res: \n{}".format(res))

        self.assertTrue( np.allclose(res.data, np.cumsum(base_series)) )

class ConvolutionMethodsTest(BasicTest):
    @testname("convolution via matrix or directly should yield equal results")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_equality(self):
        for length,filter in zip(
            CONVOLUTION_MATRIX_TEST_DATA.get("length"),
            CONVOLUTION_MATRIX_TEST_DATA.get("filter"),
            ):
            # create matrix
            matrix = delag.convolve.convolution_matrix_fixed_filter(
                length=length, filter=filter)
            # create series
            x = np.array(range(length))
            # convolve via matrix
            conv_matrix = np.dot(matrix,x)
            # convolve via np.convolve
            conv_direct = delag.convolve.convolve( x = x, filter = filter )
            # value should equal expected value
            self.assertTrue(np.allclose(conv_matrix, conv_direct))


class KernelOptimizerTest(BasicTest):
    @testname("residual with constant kernel")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_residual_constant_kernel(self):
        n = 10
        base_range = np.array(list(range(0,n))) 
        base_series   = pd.Series(
            data  = base_range,
            index = pd.DatetimeIndex(base_range * 10, dtype='datetime64[ns]'))
        self.logger.debug("base_series: \n{}".format(base_series))

        target_selection = (1,2,5,7,8)
        target_range = base_range.copy()[target_selection,]
        target_series = pd.Series(
            data  = target_range + 5,
            index = pd.DatetimeIndex(target_range * 10 - 5, 
                dtype='datetime64[ns]'))

        self.logger.debug("target_series: \n{}".format(target_series))
        optimizer = delag.optimizer.KernelOptimizer( 
            original = base_series,
            lagged   = target_series,
            kernel   = delag.kernels.LagKernelConstant( parameters = [1], 
                max_lag=1 )
            )

        expected_convolved_values = np.cumsum( base_series.values )
        self.logger.debug("expected_convolved_values: \n{}".format(
            expected_convolved_values))
        expected_inter_values = scipy.interpolate.interp1d( 
            x = base_series.index.values.astype('float64') * 1e-9,
            y = expected_convolved_values,
            kind = "linear"
            )(target_series.index.values.astype('float64') * 1e-9)
        self.logger.debug("expected_inter_values: \n{}".format(
            expected_inter_values))
        expected = np.sqrt(np.mean(
            (target_series.values - expected_inter_values) ** 2
            ))
        self.logger.debug("expected: \n{}".format(expected))
        residual = optimizer.residual
        self.logger.debug("optimizer.residual: \n{}".format(residual))
        # should be equal
        self.assertTrue( np.allclose( expected, residual ) )

    @testname("optimize with constant kernel")
    @unittest.skipIf(SKIPALL,"skipping all tests")
    def test_optimize_constant_kernel(self):
        n = 10
        base_range = np.array(list(range(0,n))) 
        base_series   = pd.Series(
            data  = base_range,
            index = pd.DatetimeIndex(base_range * 10, dtype='datetime64[ns]'))
        self.logger.debug("base_series: \n{}".format(base_series))

        target_series = pd.Series(
            data = np.cumsum(base_series.values),
            index = base_series.index.copy(),
            )
        self.logger.debug("target_series: \n{}".format(target_series))

        optimizer = delag.optimizer.KernelOptimizer( 
            original = base_series,
            lagged   = target_series,
            kernel   = delag.kernels.LagKernelConstant( parameters = [2], 
                max_lag=1 )
            )

        optimizer.optimize(method="l-bfgs-b") # optimize

        # parameter should be 1
        self.assertTrue( np.allclose( optimizer.kernel.parameters[0], 1 ) )

def run():
    # run the tests
    logger.info("=== DELAG TESTS ===")
    unittest.main(exit=False,module=__name__)
    logger.info("=== END OF DELAG TESTS ===")
