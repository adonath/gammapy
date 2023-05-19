# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Common fit statistics used in gamma-ray astronomy.

see :ref:`fit-statistics`
"""
import collections
import numpy as np
from gammapy.stats.fit_statistics_cython import TRUNCATION_VALUE
from .fit_statistics_cython import cash_sum_cython

__all__ = ["cash", "cstat", "wstat", "get_wstat_mu_bkg", "get_wstat_gof_terms"]


class FitStatistic:
    """Fits statistic base class."""

    pass


class PriorFitStatistic(FitStatistic):
    # collect all the priors from Models and evaluate them
    def __init__(self, models):
        self.models = models

    def stat_sum(self, parameters=None):
        if parameters:
            self.models.parameters = parameters

        stat_prior = 0

        for model in self.models:
            stat_prior += model.prior_log_prob(parameters)

        return stat_prior


class FitStatistic:
    """Joint fit statistic."""

    def __init__(self, statistics, weights=None):
        self.statistics = statistics

        if weights is None:
            weights = np.ones(len(statistics))

        self.weights = weights

    def stat_sum(self, parameters):
        """"""
        stat = 0

        for stat, weight in zip(self.statistics, self.weights):
            stat += stat.stat_sum(parameters) * weight

        return stat


# Maybe we have...
class DistributedJointFitStatistic:
    pass


# TODO: call this diffrently FitComponent?
class CashFitStatistic(FitStatistic):
    """Cash fit statistic."""

    def __init__(self, counts, npred_models, mask=None):
        if not counts.geom == npred_models.geom:
            raise ValueError("Counts and npred_models must have the same geometry")

        self.counts = counts
        self.npred_model = npred_models

        if mask is not None and counts.shape != mask.shape:
            raise ValueError("Mask shape does not match counts shape")

        self.mask = mask

    @classmethod
    def from_map_dataset(cls, models, dataset):
        """"""
        from gammapy.modeling.models.npred import NPredModels

        npred_models = NPredModels.from_map_dataset(models=models, dataset=dataset)
        counts = dataset.counts
        return cls(counts=counts, npred_models=npred_models)

    def peek(self):
        # Plot counts, npred and residuals
        raise NotImplementedError

    # It seems unitiutive to have those here, but all the infomation is there!
    def plot_residuals(self):
        # Copy functionality from MapDataset.plot_residuals()
        raise NotImplementedError

    def residuals(self, method="diff"):
        raise NotImplementedError

    # passing parameters is probabay only needed for the fit interface
    def stat_array(self, parameters=None):
        raise NotImplementedError

    def stat_sum(self, parameters=None, dataset=None):
        """Total statistic function value given the current model parameters."""
        counts = self.counts.data.astype(float)
        npred = self.npred_model.evaluate(
            parameters=parameters, dataset=dataset
        ).data.astype(float)

        if self.mask is not None:
            return cash_sum_cython(counts[self.mask.data], npred[self.mask.data])
        else:
            return cash_sum_cython(counts.ravel(), npred.ravel())

    def info_dict(self):
        pass


# Or alternatively
class FitStatisticContext:
    """Fit statistic handler."""

    def __init__(self, counts, npred_models, mask=None, stat=None):
        self.counts = counts
        self.npred_models = npred_models
        self.mask = mask
        self.stat = stat

    @classmethod
    def from_dataset(cls, models, dataset):
        """"""
        npred_model = NPredModels.from_dataset(models, dataset)
        counts = dataset.counts
        return cls(counts=counts, cpunts=npred_model)

    def peek(self):
        pass

    # It seems unitiutive to have thos here, but all the infomation is there!
    def plot_residuals(self):
        # Copy functionality from MapDataset.plot_residuals()
        pass

    def residuals(self):
        pass

    def stat_array(self, parameters=None):
        self.stat.evaluate()

    def stat_sum(self, parameters=None):
        pass

    @classmethod
    def from_dataset(cls, models, dataset):
        """"""
        npred_model = NPredModels.from_dataset(models, dataset)
        counts = dataset.counts
        return cls(counts=counts, cpunts=npred_model)

    def __enter__(self, dataset):
        # maybe handle IRF updates here?
        pass


class WStatFitStatistic(CashFitStatistic):
    """"""

    def __init__(self, counts, counts_off, acceptance, acceptance_off, mask=None):
        self.counts = counts
        self.counts_off = counts_off
        self.acceptance = acceptance
        self.acceptance_off = acceptance_off

    @classmethod
    def from_dataset_on_off(cls, models, dataset):
        return cls()

    def alpha(self):
        pass

    def peek(self):
        pass

    def etc(self):
        # Copy all the relevant functionality from MapDatasetOnOff
        pass


class Chi2FitStatistic(FitStatistic):
    """"""

    def __init__(self, flux_points, models, mask=None):
        self.flux_points = flux_points
        self.models = models
        self.mask = mask

    def stat_sum(self, parameters=None):
        return super().stat_sum(parameters)


def cash(n_on, mu_on, truncation_value=TRUNCATION_VALUE):
    r"""Cash statistic, for Poisson data.

    The Cash statistic is defined as:

    .. math::
        C = 2 \left( \mu_{on} - n_{on} \log \mu_{on} \right)

    and :math:`C = 0` where :math:`\mu <= 0`.
    For more information see :ref:`fit-statistics`

    Parameters
    ----------
    n_on : array_like
        Observed counts
    mu_on : array_like
        Expected counts
    truncation_value : array_like
        Minimum value use for ``mu_on``
        ``mu_on`` = ``truncation_value`` where ``n_on`` <= ``truncation_value``
        Default is 1e-25.

    Returns
    -------
    stat : ndarray
        Statistic per bin

    References
    ----------
    * `Sherpa statistics page section on the Cash statistic
      <http://cxc.cfa.harvard.edu/sherpa/statistics/#cash>`_
    * `Sherpa help page on the Cash statistic
      <http://cxc.harvard.edu/sherpa/ahelp/cash.html>`_
    * `Cash 1979, ApJ 228, 939
      <https://ui.adsabs.harvard.edu/abs/1979ApJ...228..939C>`_
    """
    n_on = np.asanyarray(n_on)
    mu_on = np.asanyarray(mu_on)
    truncation_value = np.asanyarray(truncation_value)
    if np.any(truncation_value) <= 0:
        raise ValueError("Cash statistic truncation value must be positive.")

    mu_on = np.where(mu_on <= truncation_value, truncation_value, mu_on)

    # suppress zero division warnings, they are corrected below
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = 2 * (mu_on - n_on * np.log(mu_on))
    return stat


def cstat(n_on, mu_on, truncation_value=TRUNCATION_VALUE):
    r"""C statistic, for Poisson data.

    The C statistic is defined as

    .. math::
        C = 2 \left[ \mu_{on} - n_{on} + n_{on}
            (\log(n_{on}) - log(\mu_{on}) \right]

    and :math:`C = 0` where :math:`\mu_{on} <= 0`.

    ``truncation_value`` handles the case where ``n_on`` or ``mu_on`` is 0 or less and
    the log cannot be taken.
    For more information see :ref:`fit-statistics`

    Parameters
    ----------
    n_on : array_like
        Observed counts
    mu_on : array_like
        Expected counts
    truncation_value : array_like
        ``n_on`` = ``truncation_value`` where ``n_on`` <= ``truncation_value.``
        ``mu_on`` = ``truncation_value`` where ``n_on`` <= ``truncation_value``
        Default is 1e-25.

    Returns
    -------
    stat : ndarray
        Statistic per bin

    References
    ----------
    * `Sherpa stats page section on the C statistic
      <http://cxc.cfa.harvard.edu/sherpa/statistics/#cstat>`_
    * `Sherpa help page on the C statistic
      <http://cxc.harvard.edu/sherpa/ahelp/cash.html>`_
    * `Cash 1979, ApJ 228, 939
      <https://ui.adsabs.harvard.edu/abs/1979ApJ...228..939C>`_
    """
    n_on = np.asanyarray(n_on, dtype=np.float64)
    mu_on = np.asanyarray(mu_on, dtype=np.float64)
    truncation_value = np.asanyarray(truncation_value, dtype=np.float64)

    if np.any(truncation_value) <= 0:
        raise ValueError("Cstat statistic truncation value must be positive.")

    n_on = np.where(n_on <= truncation_value, truncation_value, n_on)
    mu_on = np.where(mu_on <= truncation_value, truncation_value, mu_on)

    term1 = np.log(n_on) - np.log(mu_on)
    stat = 2 * (mu_on - n_on + n_on * term1)
    stat = np.where(mu_on > 0, stat, 0)

    return stat


def wstat(n_on, n_off, alpha, mu_sig, mu_bkg=None, extra_terms=True):
    r"""W statistic, for Poisson data with Poisson background.

    For a definition of WStat see :ref:`wstat`. If ``mu_bkg`` is not provided
    it will be calculated according to the profile likelihood formula.

    Parameters
    ----------
    n_on : array_like
        Total observed counts
    n_off : array_like
        Total observed background counts
    alpha : array_like
        Exposure ratio between on and off region
    mu_sig : array_like
        Signal expected counts
    mu_bkg : array_like, optional
        Background expected counts
    extra_terms : bool, optional
        Add model independent terms to convert stat into goodness-of-fit
        parameter, default: True

    Returns
    -------
    stat : ndarray
        Statistic per bin

    References
    ----------
    * `Habilitation M. de Naurois, p. 141
      <http://inspirehep.net/record/1122589/files/these_short.pdf>`_
    * `XSPEC page on Poisson data with Poisson background
      <https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html>`_
    """
    # Note: This is equivalent to what's defined on the XSPEC page under the
    # following assumptions
    # t_s * m_i = mu_sig
    # t_b * m_b = mu_bkg
    # t_s / t_b = alpha

    n_on = np.asanyarray(n_on, dtype=np.float64)
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)
    mu_sig = np.asanyarray(mu_sig, dtype=np.float64)

    if mu_bkg is None:
        mu_bkg = get_wstat_mu_bkg(n_on, n_off, alpha, mu_sig)

    term1 = mu_sig + (1 + alpha) * mu_bkg

    # suppress zero division warnings, they are corrected below
    with np.errstate(divide="ignore", invalid="ignore"):
        # This is a false positive error from pylint
        # See https://github.com/PyCQA/pylint/issues/2436
        term2_ = -n_on * np.log(
            mu_sig + alpha * mu_bkg
        )  # pylint:disable=invalid-unary-operand-type
    # Handle n_on == 0
    condition = n_on == 0
    term2 = np.where(condition, 0, term2_)

    # suppress zero division warnings, they are corrected below
    with np.errstate(divide="ignore", invalid="ignore"):
        # This is a false positive error from pylint
        # See https://github.com/PyCQA/pylint/issues/2436
        term3_ = -n_off * np.log(mu_bkg)  # pylint:disable=invalid-unary-operand-type
    # Handle n_off == 0
    condition = n_off == 0
    term3 = np.where(condition, 0, term3_)

    stat = 2 * (term1 + term2 + term3)

    if extra_terms:
        stat += get_wstat_gof_terms(n_on, n_off)

    return stat


def get_wstat_mu_bkg(n_on, n_off, alpha, mu_sig):
    """Background estimate ``mu_bkg`` for WSTAT.

    See :ref:`wstat`.
    """
    n_on = np.asanyarray(n_on, dtype=np.float64)
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)
    mu_sig = np.asanyarray(mu_sig, dtype=np.float64)

    # NOTE: Corner cases in the docs are all handled correctly by this formula
    C = alpha * (n_on + n_off) - (1 + alpha) * mu_sig
    D = np.sqrt(C**2 + 4 * alpha * (alpha + 1) * n_off * mu_sig)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_bkg = (C + D) / (2 * alpha * (alpha + 1))

    return mu_bkg


def get_wstat_gof_terms(n_on, n_off):
    """Goodness of fit terms for WSTAT.

    See :ref:`wstat`.
    """
    term = np.zeros(n_on.shape)

    # suppress zero division warnings, they are corrected below
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = -n_on * (1 - np.log(n_on))
        term2 = -n_off * (1 - np.log(n_off))

    term += np.where(n_on == 0, 0, term1)
    term += np.where(n_off == 0, 0, term2)

    return 2 * term
