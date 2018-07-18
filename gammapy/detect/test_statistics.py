# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to compute TS images."""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from time import time
import warnings
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from astropy.convolution import Model2DKernel, Gaussian2DKernel, CustomKernel, Kernel2D
from astropy.convolution.kernels import _round_up_to_odd_integer
from astropy.io import fits
from ..utils.array import shape_2N, symmetric_crop_pad_width
from ..irf import multi_gauss_psf_kernel
from ..maps import WcsNDMap
from ..image.models import SkyShell
from ._test_statistics_cython import (_cash_cython, _amplitude_bounds_cython,
                                      _cash_sum_cython, _f_cash_root_cython,
                                      _x_best_leastsq)

__all__ = [
    'compute_ts_image_multiscale',
    'compute_maximum_ts_image',
    'TSMapEstimator',
]

log = logging.getLogger(__name__)

FLUX_FACTOR = 1e-12
MAX_NITER = 20
RTOL = 1e-3


def _extract_array(array, shape, position):
    """Helper function to extract parts of a larger array.

    Simple implementation of an array extract function , because
    `~astropy.ndata.utils.extract_array` introduces too much overhead.`

    Parameters
    ----------
    array : `~numpy.ndarray`
        The array from which to extract.
    shape : tuple or int
        The shape of the extracted array.
    position : tuple of numbers or number
        The position of the small array's center with respect to the
        large array.
    """
    x_width = shape[1] // 2
    y_width = shape[0] // 2
    y_lo = position[0] - y_width
    y_hi = position[0] + y_width + 1
    x_lo = position[1] - x_width
    x_hi = position[1] + x_width + 1
    return array[y_lo:y_hi, x_lo:x_hi]


def f_cash(x, counts, background, model):
    """Wrapper for cash statistics, that defines the model function.

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : `~numpy.ndarray`
        Count image slice, where model is defined.
    background : `~numpy.ndarray`
        Background image slice, where model is defined.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """
    return _cash_sum_cython(counts, background + x * FLUX_FACTOR * model)


class TSMapEstimator(object):
    """
    Compute TS map using different optimization methods.

    Parameters
    ----------
    method : str ('root')
        The following options are available:

        * ``'root brentq'`` (default)
            Fit amplitude finding roots of the the derivative of
            the fit statistics. Described in Appendix A in Stewart (2009).
        * ``'root newton'``
            TODO: document
        * ``'leastsq iter'``

    error_method : ['covar', 'conf']
        Error estimation method.
    error_sigma : int (1)
        Sigma for flux error.
    ul_method : ['covar', 'conf']
        Upper limit estimation method.
    ul_sigma : int (2)
        Sigma for flux upper limits.
    n_jobs : int
        Number of parallel jobs to use for the computation.
    threshold : float (None)
        If the TS value corresponding to the initial flux estimate is not above
        this threshold, the optimizing step is omitted to save computing time.
    rtol : float (0.001)
        Relative precision of the flux estimate. Used as a stopping criterion for
        the amplitude fit.

    Notes
    -----
    Negative :math:`TS` values are defined as following:

    .. math::

        TS = \\left \\{
                 \\begin{array}{ll}
                   -TS & : \\textnormal{if} \\ F < 0 \\\\
                   \\ \\ TS & : \\textnormal{else}
                 \\end{array}
               \\right.

    Where :math:`F` is the fitted flux amplitude.

    References
    ----------
    [Stewart2009]_
    """

    def __init__(self, method='root brentq', error_method='covar', error_sigma=1,
                 ul_method='covar', ul_sigma=2, n_jobs=1, threshold=None, rtol=0.001):

        if method not in ['root brentq', 'root newton', 'leastsq iter']:
            raise ValueError("Not a valid method: '{}'".format(method))

        if error_method not in ['covar', 'conf']:
            raise ValueError("Not a valid error method '{}'".format(error_method))

        p = OrderedDict()
        p['method'] = method
        p['error_method'] = error_method
        p['error_sigma'] = error_sigma
        p['ul_method'] = ul_method
        p['ul_sigma'] = ul_sigma
        p['n_jobs'] = n_jobs
        p['threshold'] = threshold
        p['rtol'] = rtol
        self.parameters = p

    @staticmethod
    def flux_default(maps, kernel):
        """Estimate default flux map using a given kernel.

        Parameters
        ----------
        maps : `OrderedDict` or `Dict`
            Dict of input sky maps. Requires `counts`, `background` and `exposure` maps.
        kernel : `astropy.convolution.Kernel2D`
            Source model kernel.

        Returns
        -------
        flux_approx : `WcsNDMap`
            Approximate flux map.
        """
        from scipy.signal import fftconvolve
        flux = (maps['counts'].data - maps['background'].data) / maps['exposure'].data
        flux = fftconvolve(flux, kernel.array, mode='same')
        flux /= np.sum(kernel.array ** 2)
        return WcsNDMap(maps['counts'].geom, data=flux)

    @staticmethod
    def mask_default(maps, kernel):
        """Compute default mask where to estimate TS values.

        Parameters
        ----------
        maps : `OrderedDict` or `Dict`
            Dict of input sky maps. . Requires `background` and `exposure`.
        kernel : `astropy.convolution.Kernel2D`
            Source model kernel.

        Returns
        -------
        mask : : `WcsNDMap`
            Mask map.
        """
        mask = np.zeros(maps['exposure'].data.shape, dtype=int)

        # mask boundary
        slice_x = slice(kernel.shape[1] // 2, -kernel.shape[1] // 2 + 1)
        slice_y = slice(kernel.shape[0] // 2, -kernel.shape[0] // 2 + 1)
        mask[slice_y, slice_x] = 1

        # positions where exposure == 0 are not processed
        mask &= (maps['exposure'].data > 0)

        # in some image there are pixels, which have exposure, but zero
        # background, which doesn't make sense and causes the TS computation
        # to fail, this is a temporary fix
        mask[maps['background'].data == 0] = 0
        return WcsNDMap(geom= maps['exposure'].geom, data=mask, dtype='int')

    @staticmethod
    def sqrt_ts(map_ts):
        """Compute sqrt(TS) map.

        Compute sqrt(TS) as defined by:

        .. math::

            \sqrt{TS} = \\left \\{
                        \\begin{array}{ll}
                        -\sqrt{-TS} & : \\textnormal{if} \\ TS < 0 \\\\
                       \\ \\ \sqrt{TS} & : \\textnormal{else}
                     \\end{array}
                   \\right.

        Parameters
        ----------
        map_ts : `WcsNDMap`
            Input TS map.

        Returns
        -------
        sqrt_ts : `WcsNDMap`
            Sqrt(TS) map.
        """
        with np.errstate(invalid='ignore', divide='ignore'):
            ts = map_ts.data
            sqrt_ts = np.where(ts > 0, np.sqrt(ts), -np.sqrt(-ts))
        return WcsNDMap(geom=map_ts.geom, data=sqrt_ts)

    def run(self, maps, kernel, which='all'):
        """
        Run TS map estimation.

        Requires `counts`, `exposure` and `background` map to run.

        Parameters
        ----------
        kernel : `astropy.convolution.Kernel2D` or 2D `~numpy.ndarray`
            Source model kernel.
        maps : `OrderedDict`
            List of input sky maps.
        which : list of str or 'all'
            Which maps to compute.

        Returns
        -------
        maps : `OrderedDict`
            Result maps.
        """
        p = self.parameters

        if not isinstance(kernel, Kernel2D):
            kernel = CustomKernel(kernel)

        if which == 'all':
            which = ['ts', 'sqrt_ts', 'flux', 'flux_err', 'flux_ul', 'niter']

        result = OrderedDict()
        for name in which:
            data = np.nan * np.ones_like(maps['counts'].data)
            result[name] = WcsNDMap(geom=maps['counts'].geom, data=data)

        mask = self.mask_default(maps, kernel)

        if 'mask' in maps:
            mask.data &= maps['mask'].data

        if p['method'] == 'root newton':
            flux = self.flux_default(maps, kernel).data
        else:
            flux = None

        # prepare dtype for cython methods
        counts = maps['counts'].data.astype(float)
        background = maps['background'].data.astype(float)
        exposure = maps['exposure'].data.astype(float)

        # Compute null statistics per pixel for the whole image
        c_0 = _cash_cython(counts, background)

        error_method = p['error_method'] if 'flux_err' in which else 'none'
        ul_method = p['ul_method'] if 'flux_ul' in which else 'none'

        wrap = partial(
            _ts_value,
            counts=counts,
            exposure=exposure,
            background=background,
            c_0=c_0,
            kernel=kernel,
            flux=flux,
            method=p['method'],
            error_method=error_method,
            threshold=p['threshold'],
            error_sigma=p['error_sigma'],
            ul_method=ul_method,
            ul_sigma=p['ul_sigma'],
            rtol=p['rtol']
            )


        x, y = np.where(mask.data)
        positions = list(zip(x, y))

        with Pool(processes=p['n_jobs']) as pool:
            log.info('Using {} jobs to compute TS map.'.format(p['n_jobs']))
            results = pool.map(wrap, positions)

        # Set TS values at given positions
        j, i = zip(*positions)
        for name in ['ts', 'flux', 'niter']:
            result[name].data[j, i] = [_[name] for _ in results]

        if 'flux_err' in which:
            result['flux_err'].data[j, i] = [_['flux_err'] for _ in results]

        if 'flux_ul' in which:
            result['flux_ul'].data[j, i] = [_['flux_ul'] for _ in results]

        # Compute sqrt(TS) values
        if 'sqrt_ts' in which:
            result['sqrt_ts'] = self.sqrt_ts(result['ts'])

        return result

    def __str__(self):
        """
        Info string.
        """
        p = self.parameters
        info = self.__class__.__name__
        info += '\n\nParameters:\n\n'
        for key in p:
            info += '\t{key:13s}: {value}\n'.format(key=key, value=p[key])
        return info


def compute_ts_image_multiscale(maps, psf_parameters, scales=[0], downsample='auto',
                                residual=False, morphology='Gaussian2D', width=None,
                                **kwargs):
    """Compute multi-scale TS images using ``compute_ts_image``.

    High level TS image computation using a multi-Gauss PSF kernel and assuming
    a given source morphology. To optimize the performance the input data
    can be sampled down when computing TS images on larger scales.

    Parameters
    ----------
    images : `~gammapy.image.SkyImageList`
        Image collection containing the data. Must contain the following:
            * 'counts', Counts image
            * 'background', Background image
            * 'exposure', Exposure image
    psf_parameters : dict
        Dict defining the multi gauss PSF parameters.
        See `~gammapy.irf.multi_gauss_psf` for details.
    scales : list ([0])
        List of scales to use for TS image computation.
    downsample : int ('auto')
        Down sampling factor. Can be set to 'auto' if the down sampling
        factor should be chosen automatically.
    residual : bool (False)
        Compute a TS residual image.
    morphology : str ('Gaussian2D')
        Source morphology assumption. Either 'Gaussian2D' or 'Shell2D'.
    **kwargs : dict
        Keyword arguments forwarded to `TSImageEstimator`.

    Returns
    -------
    multiscale_result : list
        List of `~gammapy.image.SkyImageList` objects.
    """
    BINSZ = maps['counts'].geom.pixel_scales.mean().value
    shape = maps['counts'].data.shape

    multiscale_result = []

    ts_estimator = TSMapEstimator(**kwargs)

    for scale in scales:
        log.info('Computing {}TS image for scale {:.3f} deg and {}'
                 ' morphology.'.format('residual ' if residual else '',
                                       scale,
                                       morphology))

        # Sample down and require that scale parameters is at least 5 pix
        if downsample == 'auto':
            factor = int(np.select([scale < 5 * BINSZ, scale < 10 * BINSZ,
                                    scale < 20 * BINSZ, scale < 40 * BINSZ],
                                   [1, 2, 4, 4], 8))
        else:
            factor = int(downsample)

        if factor == 1:
            log.info('No down sampling used.')
            downsampled = False
        else:
            if morphology == 'Shell2D':
                factor /= 2
            log.info('Using down sampling factor of {}'.format(factor))
            downsampled = True

        maps_downsampled = OrderedDict()
        for name in maps:
            if downsampled:
                pad_width = symmetric_crop_pad_width(shape, shape_2N(shape))[0]
                maps_downsampled[name] = maps[name].pad(pad_width)
                preserve_counts = name in ['counts', 'background', 'exclusion']
                maps_downsampled[name] = maps_downsampled[name].downsample(factor,
                                             preserve_counts=preserve_counts)
            else:
                maps_downsampled[name] = maps[name]

        # Set up PSF and source kernel
        kernel = multi_gauss_psf_kernel(psf_parameters, BINSZ=BINSZ,
                                        NEW_BINSZ=BINSZ * factor,
                                        mode='oversample')

        if scale > 0:
            from astropy.convolution import convolve
            sigma = scale / (BINSZ * factor)
            if morphology == 'Gaussian2D':
                source_kernel = Gaussian2DKernel(sigma, mode='oversample')
            elif morphology == 'Shell2D':
                model = SkyShell(lon_0=0, lat_0=0, r_i=sigma, r_o=(1 + width) * sigma)
                x_size = _round_up_to_odd_integer(2 * sigma * (1 + width) + kernel.shape[0] / 2)
                source_kernel = Model2DKernel(model, x_size=x_size, mode='oversample')
            else:
                raise ValueError('Unknown morphology: {}'.format(morphology))
            kernel = convolve(source_kernel, kernel)
            kernel.normalize()

        if residual:
            maps_downsampled['background'].data += maps_downsampled['model'].data

        # Compute TS image
        ts_results = ts_estimator.run(maps_downsampled, kernel)

        if downsampled:
            for name, order in zip(['ts', 'sqrt_ts', 'flux', 'flux_err', 'niter'], [1, 1, 1, 0]):
                preserve_counts = name in ['flux', 'flux_err']
                ts_results[name] = ts_results[name].upsample(
                    factor=factor,
                    preserve_counts=preserve_counts,
                    order=order
                    )
                ts_results[name] = ts_results[name].crop(crop_width=pad_width)

        multiscale_result.append(ts_results)

    return multiscale_result


def compute_maximum_ts_image(ts_image_results):
    """Compute maximum TS image across a list of given TS images.

    Parameters
    ----------
    ts_image_results : list
        List of `~gammapy.image.SkyImageList` objects.

    Returns
    -------
    images : `~gammapy.image.SkyImageList`
        Images (ts, niter, amplitude)
    """
    # Get data
    ts = np.dstack([result.ts for result in ts_image_results])
    niter = np.dstack([result.niter for result in ts_image_results])
    amplitude = np.dstack([result.amplitude for result in ts_image_results])
    scales = [result.scale for result in ts_image_results]

    # Set up max arrays
    ts_max = np.max(ts, axis=2)
    scale_max = np.zeros(ts.shape[:-1])
    niter_max = np.zeros(ts.shape[:-1])
    amplitude_max = np.zeros(ts.shape[:-1])

    for idx_scale, scale in enumerate(scales):
        index = np.where(ts[:, :, idx_scale] == ts_max)
        scale_max[index] = scale
        niter_max[index] = niter[:, :, idx_scale][index]
        amplitude_max[index] = amplitude[:, :, idx_scale][index]

    meta = OrderedDict()
    meta['MORPH'] = (ts_image_results[0].morphology, 'Source morphology assumption')

    return SkyImageList([
        SkyImage(name='ts', data=ts_max.astype('float32')),
        SkyImage(name='niter', data=niter_max.astype('int16')),
        SkyImage(name='amplitude', data=amplitude_max.astype('float32')),
    ], meta=meta)


def _ts_value(position, counts, exposure, background, c_0, kernel, flux,
              method, error_method, error_sigma, ul_method, ul_sigma, threshold, rtol):
    """Compute TS value at a given pixel position.

    Uses approach described in Stewart (2009).

    Parameters
    ----------
    position : tuple (i, j)
        Pixel position.
    counts : `~numpy.ndarray`
        Counts image
    background : `~numpy.ndarray`
        Background image
    exposure : `~numpy.ndarray`
        Exposure image
    kernel : `astropy.convolution.Kernel2D`
        Source model kernel
    flux : `~numpy.ndarray`
        Flux image. The flux value at the given pixel position is used as
        starting value for the minimization.

    Returns
    -------
    TS : float
        TS value at the given pixel position.
    """
    # Get data slices
    counts_ = _extract_array(counts, kernel.shape, position)
    background_ = _extract_array(background, kernel.shape, position)
    exposure_ = _extract_array(exposure, kernel.shape, position)
    c_0_ = _extract_array(c_0, kernel.shape, position)

    model = (exposure_ * kernel._array)

    c_0 = c_0_.sum()

    if threshold is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            c_1 = f_cash(flux[position], counts_, background_, model)
        # Don't fit if pixel significance is low
        if c_0 - c_1 < threshold:
            return c_0 - c_1, flux[position] * FLUX_FACTOR, 0

    if method == 'root brentq':
        amplitude, niter = _root_amplitude_brentq(counts_, background_, model, rtol=rtol)
    elif method == 'root newton':
        amplitude, niter = _root_amplitude(counts_, background_, model, flux[position], rtol=rtol)
    elif method == 'leastsq iter':
        amplitude, niter = _leastsq_iter_amplitude(counts_, background_, model, rtol=rtol)
    else:
        raise ValueError('Invalid method: {}'.format(method))

    with np.errstate(invalid='ignore', divide='ignore'):
        c_1 = f_cash(amplitude, counts_, background_, model)

    result = {}
    result['ts'] = (c_0 - c_1) * np.sign(amplitude)
    result['flux'] = amplitude * FLUX_FACTOR
    result['niter'] = niter

    if error_method == 'covar':
        flux_err = _compute_flux_err_covar(amplitude, counts_, background_, model)
        result['flux_err'] = flux_err * error_sigma
    elif error_method == 'conf':
        flux_err = _compute_flux_err_conf(amplitude, counts_, background_, model,
                                          c_1, error_sigma)
        result['flux_err'] = FLUX_FACTOR * flux_err

    if ul_method == 'covar':
        result['flux_ul'] = result['flux'] + ul_sigma * result['flux_err']
    elif ul_method == 'conf':
        flux_ul = _compute_flux_err_conf(amplitude, counts_, background_, model,
                                         c_1, ul_sigma)
        result['flux_ul'] = FLUX_FACTOR * flux_ul + result['flux']
    return result


def _leastsq_iter_amplitude(counts, background, model, maxiter=MAX_NITER, rtol=RTOL):
    """Fit amplitude using an iterative least squares algorithm.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Slice of counts image
    background : `~numpy.ndarray`
        Slice of background image
    model : `~numpy.ndarray`
        Model template to fit.
    maxiter : int
        Maximum number of iterations.
    rtol : float
        Relative flux error.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    bounds = _amplitude_bounds_cython(counts, background, model)
    amplitude_min, amplitude_max, amplitude_min_total = bounds

    if not counts.sum() > 0:
        return amplitude_min_total, 0

    weights = np.ones(model.shape)

    x_old = 0
    for i in range(maxiter):
        x = _x_best_leastsq(counts, background, model, weights)
        if abs((x - x_old) / x) < rtol:
            return max(x / FLUX_FACTOR, amplitude_min_total), i + 1
        else:
            weights = x * model + background
            x_old = x
    return max(x / FLUX_FACTOR, amplitude_min_total), MAX_NITER


def _root_amplitude(counts, background, model, flux, rtol=RTOL):
    """Fit amplitude by finding roots using newton algorithm.

    See Appendix A Stewart (2009).

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Slice of count image
    background : `~numpy.ndarray`
        Slice of background image
    model : `~numpy.ndarray`
        Model template to fit.
    flux : float
        Starting value for the fit.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    from scipy.optimize import newton

    args = (counts, background, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return newton(_f_cash_root_cython, flux, args=args, maxiter=MAX_NITER, tol=rtol), 0
        except RuntimeError:
            # Where the root finding fails NaN is set as amplitude
            return np.nan, MAX_NITER


def _root_amplitude_brentq(counts, background, model, rtol=RTOL):
    """Fit amplitude by finding roots using Brent algorithm.

    See Appendix A Stewart (2009).

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Slice of count image
    background : `~numpy.ndarray`
        Slice of background image
    model : `~numpy.ndarray`
        Model template to fit.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    from scipy.optimize import brentq

    # Compute amplitude bounds and assert counts > 0
    bounds = _amplitude_bounds_cython(counts, background, model)
    amplitude_min, amplitude_max, amplitude_min_total = bounds

    if not counts.sum() > 0:
        return amplitude_min_total, 0

    args = (counts, background, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = brentq(_f_cash_root_cython, amplitude_min, amplitude_max, args=args,
                            maxiter=MAX_NITER, full_output=True, rtol=rtol)
            return max(result[0], amplitude_min_total), result[1].iterations
        except (RuntimeError, ValueError):
            # Where the root finding fails NaN is set as amplitude
            return np.nan, MAX_NITER


def _compute_flux_err_covar(x, counts, background, model):
    """
    Compute amplitude errors using inverse 2nd derivative method.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        stat = (model ** 2 * counts) / (background + x * FLUX_FACTOR * model) ** 2
        return np.sqrt(1. / stat.sum())


def _compute_flux_err_conf(amplitude, counts, background, model, c_1, error_sigma):
    """
    Compute amplitude errors using likelihood profile method.
    """
    from scipy.optimize import brentq

    def ts_diff(x, counts, background, model):
        return (c_1 + error_sigma ** 2) - f_cash(x, counts, background, model)

    args = (counts, background, model)

    amplitude_max = amplitude + 1E4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = brentq(ts_diff, amplitude, amplitude_max, args=args,
                            maxiter=MAX_NITER, rtol=1e-3)
            return result - amplitude
        except (RuntimeError, ValueError):
            # Where the root finding fails NaN is set as amplitude
            return np.nan
