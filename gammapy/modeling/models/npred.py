import collections
import os
from typing import Any
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from gammapy.maps import Map
from gammapy.modeling.models import Model
from .cube import FoVBackgroundModel

# The NPredModel will not have a global evalation mode
# either support GTI or a TimeMapAxis

USE_NPRED_CACHE = True
PSF_MAX_RADIUS = None
PSF_CONTAINMENT = 0.999
CUTOUT_MARGIN = 0.1 * u.deg
TEMPORAL_OVERSAMPLING_FACTOR = 100


def apply_edisp(input_map, edisp):
    """Apply energy dispersion to map. Requires "energy_true" axis.

    Parameters
    ----------
    input_map : `~gammapy.maps.Map`
        The map to be convolved with the energy dispersion.
        It must have an axis named "energy_true"
    edisp : `gammapy.irf.EDispKernel`
        Energy dispersion matrix

    Returns
    -------
    map : `~gammapy.maps.Map`
        Map with energy dispersion applied.
    """
    # TODO: either use sparse matrix mutiplication or something like edisp.is_diagonal
    if edisp is not None:
        loc = input_map.geom.axes.index("energy_true")
        data = np.rollaxis(input_map.data, loc, len(input_map.data.shape))
        data = np.dot(data, edisp.pdf_matrix)
        data = np.rollaxis(data, -1, loc)
        energy_axis = edisp.axes["energy"].copy(name="energy")
    else:
        data = input_map.data
        energy_axis = input_map.geom.axes["energy_true"].copy(name="energy")

    geom = input_map.geom.to_image().to_cube(axes=[energy_axis])

    return Map.from_geom(geom=geom, data=data, unit=input_map.unit)


class cache_evaluation:
    # this is a decorator that can be used to cache the results of a function call
    # it is probablay a bit more flexible than the lru_cache decorator
    # and allows for paramerter precision and other things
    # TODO: add norm caching
    def __init__(self):
        self.cached_parameter_values = None
        self.cache = None

    def __call__(self, func):
        def cache_wrapper(_, parameters, **kwargs):
            """Wrapper method"""
            if self.cached_parameter_values is not None:
                if np.allclose(parameters.value, self.cached_parameter_values):
                    return self.cache

            self.cached_parameter_values = parameters.value
            self.cache = func(self=_, parameters=parameters, **kwargs)
            return self.cache

        return cache_wrapper


class NPredModel(Model):
    """Predicted counts model class.

    Parameters
    ----------
    sky_model : `~gammapy.modeling.models.SkyModel`
        Sky model
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure map
    psf : `~gammapy.irf.PSFKernel`
        PSF map
    edisp : `~gammapy.irf.EDispKernel`
        Energy dispersion map
    mask : `~gammapy.maps.WcsNDMap`
        Mask map
    gti : `~gammapy.data.GTI`
        GTI of the observation
    """

    def __init__(self, model, exposure, psf, edisp, gti=None):
        # The sky model attribute should immutable
        self.model = model
        self.exposure = exposure
        self.psf = psf
        self.edisp = edisp
        self.gti = gti

    @property
    def name(self):
        """Model name"""
        return self.model.name

    @property
    def dataset_names(self):
        """Model name"""
        return self.model.dataset_names

    @property
    def needs_update(self):
        """Check whether the model IRFs need an update"""
        # compare the postion of the sky model against the positon of the exposure map ort IRF kernel meta
        pass

    @property
    def geom_true(self):
        """True energy map geometry (`~gammapy.maps.Geom`)"""
        return self.exposure.geom

    @property
    def geom_true_image(self):
        """True energy map geometry (`~gammapy.maps.Geom`)"""
        return self.geom_true.to_image()

    @property
    def energy_axis_true(self):
        """True energy map geometry (`~gammapy.maps.Geom`)"""
        return self.geom_true.axes["energy_true"]

    @property
    def apply_psf_after_edisp(self):
        """Whether to apply the PSF after the energy dispersion"""
        return (self.psf is not None) and self.psf.has_energy_axis

    @property
    def apply_psf(self):
        """Apply PSF"""
        return self.model.apply_irf["psf"] and self.psf is not None

    @property
    def apply_exposure(self):
        """Apply exposure"""
        return self.model.apply_irf["exposure"] and self.exposure is not None

    @property
    def apply_edisp(self):
        """Apply edisp"""
        return self.model.apply_irf["edisp"] and self.edisp is not None

    @property
    def psf_width(self):
        """Width of the PSF"""
        if self.psf is not None:
            psf_width = np.max(self.psf.psf_kernel_map.geom.width)
        else:
            psf_width = 0 * u.deg
        return psf_width

    @staticmethod
    def cutout_width(psf, model):
        """Cutout width for the model component"""
        if psf is not None:
            psf_width = np.max(psf.psf_kernel_map.geom.width)
        else:
            psf_width = 0 * u.deg

        return psf_width + 2 * (model.evaluation_radius + CUTOUT_MARGIN)

    @classmethod
    def from_map_dataset(cls, model, dataset):
        """Create NPredModel from a MapDataset

        Parameters
        ----------
        model : `~gammapy.modeling.models.SkyModel`
            Sky model
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset

        Returns
        -------
        npred_model : `~gammapy.modeling.models.NPredModel`
            Predicted counts model
        """
        geom = dataset._geom

        if dataset.edisp:
            energy_axis = geom.axes["energy"]

            edisp = dataset.edisp.get_edisp_kernel(
                position=model.position, energy_axis=energy_axis
            )
        else:
            edisp = None

        # lookup psf
        if dataset.psf and model.spatial_model:
            if dataset.psf.is_reco:
                geom_psf = geom
            else:
                geom_psf = dataset.exposure.geom

            if geom_psf.is_hpx:
                geom_psf = geom_psf.to_wcs_geom()

            psf = dataset.psf.get_psf_kernel(
                position=model.position,
                geom=geom_psf,
                containment=PSF_CONTAINMENT,
                max_radius=PSF_MAX_RADIUS,
            )
        else:
            psf = None

        width = cls.cutout_width(model=model, psf=psf)

        exposure = dataset.exposure.cutout(
            position=model.position, width=width, odd_npix=True
        )

        return cls(
            model=model, exposure=exposure, psf=psf, edisp=edisp, gti=dataset.gti
        )

    @property
    def parameters(self):
        """Model parameters"""
        return self.model.parameters

    # TODO support caching?
    def evalute_dnde(self, parameters=None):
        """Evaluate model differential flux at map pixel centers.

        Returns
        -------
        map : `~gammapy.maps.Map`
            Sky cube with data filled with evaluated model values.
            Units: ``cm-2 s-1 TeV-1 deg-2``
        """
        if parameters:
            self.model.parameters.value = parameters.value

        return self.model.evaluate_geom(geom=self.geom_true, gti=self.gti)

    @cache_evaluation()
    def evaluate_flux_spectral(self, parameters=None):
        """Compute spectral flux"""
        if parameters:
            self.model.spectral_model.parameters.value = parameters.value

        energy_min = self.energy_axis_true.edges_min
        energy_max = self.energy_axis_true.edges_max

        value = self.model.spectral_model.integral(
            energy_min=energy_min, energy_max=energy_max
        )

        if self.geom_true.is_hpx:
            shape = (-1, 1)
        else:
            shape = (-1, 1, 1)

        return value.reshape(shape)

    def evaluate_flux_spatial(self, parameters=None):
        """Evaluate spatial model"""
        if parameters:
            self.model.spatial_model.parameters.value = parameters.value

        if self.model.spatial_model.is_energy_dependent:
            geom = self.geom_true
        else:
            geom = self.geom_true_image

        return self.model.spatial_model.integrate_geom(geom=geom)

    @cache_evaluation()
    def evaluate_flux_spatial_psf(self, parameters=None):
        """Evaluate spatial model and convolve PSF"""
        flux = self.evaluate_flux_spatial(parameters=parameters)

        if self.psf and self.model.apply_irf["psf"]:
            flux = flux.convolve(self.psf)

        return flux

    @cache_evaluation()
    def evaluate_flux(self, parameters=None):
        """Compute psf convolved and temporal model corrected flux."""
        flux = self.evaluate_flux_spectral(parameters=parameters.spectral)

        if self.model.spatial_model:
            if self.apply_psf_after_edisp:
                flux_spatial = self.evaluate_flux_spatial(parameters=parameters.spatial)
            else:
                flux_spatial = self.evaluate_flux_spatial_psf(
                    parameters=parameters.spatial
                )

            flux = flux * flux_spatial

        if self.model.temporal_model:
            flux *= self.evaluate_norm_temporal(parameters=parameters.temporal)

        return Map.from_geom(geom=self.geom_true, data=flux.value, unit=flux.unit)

    def evaluate_norm_temporal(self, parameters=None):
        """Compute temporal norm"""
        if parameters:
            self.model.temporal_model.parameters.value = parameters.value

        integral = self.model.temporal_model.integral(
            t_min=self.gti.time_start,
            t_max=self.gti.time_stop,
            oversampling_factor=TEMPORAL_OVERSAMPLING_FACTOR,
        )
        return np.sum(integral)

    @cache_evaluation()
    def evaluate(self, parameters=None):
        """Evaluate model"""
        npred = self.evaluate_flux(parameters=parameters)

        if self.apply_exposure:
            npred = npred * self.exposure

        if self.apply_edisp:
            npred = apply_edisp(npred, self.edisp)

        return npred

    @cache_evaluation()
    def evaluate_psf_after_edisp(self, parameters=None):
        """Evaluate model"""
        npred = self.evaluate_flux(parameters=parameters)

        if self.apply_exposure:
            npred = npred * self.exposure

        if self.apply_edisp:
            npred = apply_edisp(npred, self.edisp)

        if self.apply_psf:
            npred = npred.convolve(self.psf)

        return npred

    def contributes(self, mask):
        """Check if the model contributes to the given mask"""
        return self.model.contributes(mask=mask, margin=self.psf_width)

    def read():
        """Read model from file."""
        # less clear what to do here, but it could be useful to quickly setup new models
        raise NotImplementedError

    def write():
        """Write model to file."""
        # less clear what to do here, but it could be useful to quickly setup new models
        # it bundles a YAML model definiton with FITS based IRFs
        raise NotImplementedError

    def peek(self, figsize=(12, 15)):
        """Quick-look summary plots.

        Parameters
        ----------
        figsize : tuple
            Size of the figure.
        """
        nrows = 1

        if self.psf:
            nrows += 1

        if self.edisp:
            nrows += 1

        fig, axes = plt.subplots(
            ncols=2,
            nrows=nrows,
            subplot_kw={"projection": self.exposure.geom.wcs},
            figsize=figsize,
            gridspec_kw={"hspace": 0.2, "wspace": 0.3},
        )

        axes = axes.flat

        axes[0].set_title("Predicted counts")
        self.evaluate().sum_over_axes().plot(ax=axes[0], add_cbar=True)

        axes[1].set_title("Exposure")
        self.exposure.sum_over_axes().plot(ax=axes[1], add_cbar=True)

        idx = 3

        if self.psf:
            axes[2].set_title("Energy-integrated PSF kernel")
            self.psf.plot_kernel(ax=axes[2], add_cbar=True)

            axes[3].set_title("PSF kernel at 1 TeV")
            self.psf.plot_kernel(ax=axes[3], add_cbar=True, energy=1 * u.TeV)

            idx += 2

        if self.edisp:
            axes[idx - 1].remove()
            ax = fig.add_subplot(nrows, 2, idx)
            ax.set_title("Energy bias")
            self.edisp.plot_bias(ax=ax)

            axes[idx].remove()
            ax = fig.add_subplot(nrows, 2, idx + 1)
            ax.set_title("Energy dispersion matrix")
            self.edisp.plot_matrix(ax=ax)


class RegionNPredModel(Model):
    # we have some special cases for the point like region base methods
    # maybe bundle those into a new RegionNPredModel class to make it cleaner.
    """"""
    pass


class TemplateNPredModel(Model):
    """Background model.

    Create a new map by a tilt and normalization on the available map

    Parameters
    ----------
    map : `~gammapy.maps.Map`
        Background model map
    model: `~FoVBackgroundModel`
        FoV background model
    """

    needs_update = False
    tag = "TemplateNPredModel"

    def __init__(self, map, model=None):
        self.map = map
        self.model = model

    @property
    def name(self):
        return self._name

    @property
    def dataset_names(self):
        """Model name"""
        return self.model.dataset_names

    def contributes(self, mask):
        """Check if the model contributes to the given mask"""
        return True

    @property
    def parameters(self):
        """Model parameters"""
        return self.model.parameters

    @property
    def energy_center(self):
        """True energy axis bin centers (`~astropy.units.Quantity`)"""
        energy_axis = self.map.geom.axes["energy"]
        energy = energy_axis.center
        return energy[:, np.newaxis, np.newaxis]

    def evaluate_norm(self, parameters=None):
        """Evaluate norm"""
        # if parameters:
        #    self.model.parameters = parameters

        return self.model.evaluate(energy=self.energy_center)

    @cache_evaluation()
    def evaluate(self, parameters=None):
        """Evaluate TemplateNPredModel.

        Returns
        -------
        npred : `~gammapy.maps.Map`
           Predicted counts map.
        """
        norm = self.evaluate_norm(parameters=parameters)
        data = self.map.data * norm
        return self.map.copy(data=data)

    def write(self, overwrite=False):
        """"""
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def peek(self):
        raise NotImplementedError


class NPredModels(collections.abc.Sequence):
    """NPredModels container class."""

    def __init__(self, npred_models, geom):
        self.npred_models = npred_models
        self.geom = geom

    def __getitem__(self, key):
        return self.npred_models[key]

    def __len__(self):
        return len(self.npred_models)

    @classmethod
    def from_map_dataset(cls, models, dataset):
        """Create from dataset"""
        # Select models here or earlier...
        npred_model = []

        for model in models:
            if isinstance(model, FoVBackgroundModel):
                model = TemplateNPredModel(
                    map=dataset.background,
                    model=model,
                )
            else:
                model = NPredModel.from_map_dataset(model=model, dataset=dataset)

            npred_model.append(model)

        return cls(npred_model, geom=dataset.counts.geom)

    def npred_signal(self, parameters=None):
        """Total predicted source and background counts

        Returns
        -------
        npred : `Map`
            Total predicted counts
        """
        # select signal only
        raise NotImplementedError

    def npred_background(self):
        """Predicted background counts

        The predicted background counts depend on the parameters
        of the `FoVBackgroundModel` defined in the dataset.

        Returns
        -------
        npred_background : `Map`
            Predicted counts from the background.
        """
        # select background only
        raise NotImplementedError

    @cache_evaluation()
    def evaluate(self, parameters=None, dataset=None):
        """Model predicted signal counts.

        If a list of model name is passed, predicted counts from these components are returned.
        If stack is set to True, a map of the sum of all the predicted counts is returned.
        If stack is set to False, a map with an additional axis representing the models is returned.

        Parameters
        ----------
        parameters : `Parameters`
            Model parameters
        dataset : `Dataset`
            Dataset

        Returns
        -------
        npred_sig: `gammapy.maps.Map`
            Map of the predicted signal counts
        """
        npred_total = Map.from_geom(geom=self.geom)

        for idx, npred_model in enumerate(self.npred_models):
            if npred_model.needs_update and dataset is not None:
                npred_model = NPredModel.from_map_dataset(
                    model=self.model, dataset=dataset
                )
                self.npred_models[idx] = npred_model

            if npred_model.contributes(mask=dataset.mask):
                # TODO:  match the parameter lists such that one can use parameters
                npred = npred_model.evaluate(parameters=npred_model.parameters)
                npred_total.stack(npred)

        npred_total.data[npred_total.data < 0.0] = 0
        return npred_total

    def cutout(self):
        """Cutout the npred models"""
        raise NotImplementedError

    def slice_by_idx(self):
        """Slice the mpred models"""
        raise NotImplementedError

    def read():
        # less clear what to do here, but it could be useful to quickly setup new fits
        pass

    def write():
        pass
