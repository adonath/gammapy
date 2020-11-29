# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Cube models (axes: lon, lat, energy)."""
import numpy as np
import astropy.units as u
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling import Covariance, Parameters
from gammapy.modeling.parameter import _get_parameters_str
from gammapy.utils.scripts import make_name, make_path
from .core import Model, Models
from .spatial import ConstantSpatialModel, SpatialModel, TemplateSpatialModel
from .spectral import PowerLawNormSpectralModel, SpectralModel, TemplateSpectralModel
from .temporal import TemporalModel

__all__ = [
    "SkyModel",
    "FoVBackgroundModel",
    "create_fermi_isotropic_diffuse_model",
]


class SkyModel(Model):
    """Sky model component.

    This model represents a factorised sky model.
    It has `~gammapy.modeling.Parameters`
    combining the spatial and spectral parameters.

    Parameters
    ----------
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Spectral model
    spatial_model : `~gammapy.modeling.models.SpatialModel`
        Spatial model (must be normalised to integrate to 1)
    temporal_model : `~gammapy.modeling.models.temporalModel`
        Temporal model
    name : str
        Model identifier
    apply_irf : dict
        Dictionary declaring which IRFs should be applied to this model. Options
        are {"exposure": True, "psf": True, "edisp": True}
    datasets_names : list of str
        Which datasets this model is applied to.
    """

    tag = ["SkyModel", "sky-model"]
    _apply_irf_default = {"exposure": True, "psf": True, "edisp": True}

    def __init__(
        self,
        spectral_model,
        spatial_model=None,
        temporal_model=None,
        name=None,
        apply_irf=None,
        datasets_names=None,
    ):
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model
        self.temporal_model = temporal_model
        self._name = make_name(name)

        if apply_irf is None:
            apply_irf = self._apply_irf_default.copy()

        self.apply_irf = apply_irf
        self.datasets_names = datasets_names
        #self._check_unit()
        super().__init__()

    @property
    def _models(self):
        models = self.spectral_model, self.spatial_model, self.temporal_model
        return [model for model in models if model is not None]

    def _check_covariance(self):
        if not self.parameters == self._covariance.parameters:
            self._covariance = Covariance.from_stack(
                [model.covariance for model in self._models],
            )

    def _check_unit(self):
        from gammapy.data.gti import GTI

        # evaluate over a test geom to check output unit
        # TODO simpler way to test this ?
        axis = MapAxis.from_energy_bounds(
            "0.1 TeV", "10 TeV", nbin=1, name="energy_true"
        )

        geom = WcsGeom.create(skydir=self.position, npix=(2, 2), axes=[axis])

        gti = GTI.create(1 * u.day, 2 * u.day)
        value = self.evaluate_geom(geom, gti)

        if self.apply_irf["exposure"]:
            ref_unit = u.Unit("cm-2 s-1 MeV-1 sr-1")
        else:
            ref_unit = u.Unit("sr-1")

        if self.spatial_model is None:
            ref_unit = ref_unit / u.Unit("sr-1")

        if not value.unit.is_equivalent(ref_unit):
            raise ValueError(
                f"SkyModel unit {value.unit} is not equivalent to {ref_unit}"
            )

    @property
    def covariance(self):
        self._check_covariance()

        for model in self._models:
            self._covariance.set_subcovariance(model.covariance)

        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        self._check_covariance()
        self._covariance.data = covariance

        for model in self._models:
            subcovar = self._covariance.get_subcovariance(model.covariance.parameters)
            model.covariance = subcovar

    @property
    def name(self):
        return self._name

    @property
    def parameters(self):
        parameters = []

        parameters.append(self.spectral_model.parameters)

        if self.spatial_model is not None:
            parameters.append(self.spatial_model.parameters)

        if self.temporal_model is not None:
            parameters.append(self.temporal_model.parameters)

        return Parameters.from_stack(parameters)

    @property
    def spatial_model(self):
        """`~gammapy.modeling.models.SpatialModel`"""
        return self._spatial_model

    @spatial_model.setter
    def spatial_model(self, model):
        if not (model is None or isinstance(model, SpatialModel)):
            raise TypeError(f"Invalid type: {model!r}")

        self._spatial_model = model

    @property
    def spectral_model(self):
        """`~gammapy.modeling.models.SpectralModel`"""
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, model):
        if not (model is None or isinstance(model, SpectralModel)):
            raise TypeError(f"Invalid type: {model!r}")
        self._spectral_model = model

    @property
    def temporal_model(self):
        """`~gammapy.modeling.models.TemporalModel`"""
        return self._temporal_model

    @temporal_model.setter
    def temporal_model(self, model):
        if not (model is None or isinstance(model, TemporalModel)):
            raise TypeError(f"Invalid type: {model!r}")

        self._temporal_model = model

    @property
    def position(self):
        """`~astropy.coordinates.SkyCoord`"""
        return getattr(self.spatial_model, "position", None)

    @property
    def evaluation_radius(self):
        """`~astropy.coordinates.Angle`"""
        return getattr(self.spatial_model, "evaluation_radius", None)

    @property
    def frame(self):
        return self.spatial_model.frame

    def __add__(self, other):
        if isinstance(other, (Models, list)):
            return Models([self, *other])
        elif isinstance(other, SkyModel):
            return Models([self, other])
        else:
            raise TypeError(f"Invalid type: {other!r}")

    def __radd__(self, model):
        return self.__add__(model)

    def __call__(self, lon, lat, energy, time=None):
        return self.evaluate(lon, lat, energy, time)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"spatial_model={self.spatial_model!r}, "
            f"spectral_model={self.spectral_model!r})"
            f"temporal_model={self.temporal_model!r})"
        )

    def evaluate(self, lon, lat, energy, time=None):
        """Evaluate the model at given points.

        The model evaluation follows numpy broadcasting rules.

        Return differential surface brightness cube.
        At the moment in units: ``cm-2 s-1 TeV-1 deg-2``

        Parameters
        ----------
        lon, lat : `~astropy.units.Quantity`
            Spatial coordinates
        energy : `~astropy.units.Quantity`
            Energy coordinate
        time: `~astropy.time.Time`
            Time co-ordinate

        Returns
        -------
        value : `~astropy.units.Quantity`
            Model value at the given point.
        """
        value = self.spectral_model(energy)  # pylint:disable=not-callable
        # TODO: case if self.temporal_model is not None, introduce time in arguments ?

        if self.spatial_model is not None:
            if self.spatial_model.is_energy_dependent:
                spatial = self.spatial_model(lon, lat, energy)
            else:
                spatial = self.spatial_model(lon, lat)

            value = value * spatial  # pylint:disable=not-callable

        if (self.temporal_model is not None) and (time is not None):
            value = value * self.temporal_model(time)

        return value

    def evaluate_geom(self, geom, gti=None):
        """Evaluate model on `~gammapy.maps.Geom`."""
        energy = geom.axes["energy_true"].center[:, np.newaxis, np.newaxis]
        value = self.spectral_model(energy)

        if self.spatial_model:
            value = value * self.spatial_model.evaluate_geom(geom)

        if self.temporal_model:
            integral = self.temporal_model.integral(gti.time_start, gti.time_stop)
            value = value * np.sum(integral)

        return value

    def integrate_geom(self, geom, gti=None):
        """Integrate model on `~gammapy.maps.Geom`.

        Parameters
        ----------
        geom : `Geom` or `~gammapy.maps.RegionGeom`
            Map geometry
        gti : `GTI`
            GIT table

        Returns
        -------
        flux : `Map`
            Predicted flux map
        """
        energy = geom.axes["energy_true"].edges
        value = self.spectral_model.integral(energy[:-1], energy[1:],).reshape(
            (-1, 1, 1)
        )

        if self.spatial_model:
            value = value * self.spatial_model.integrate_geom(geom).quantity

        if self.temporal_model:
            integral = self.temporal_model.integral(gti.time_start, gti.time_stop)
            value = value * np.sum(integral)

        return Map.from_geom(geom=geom, data=value.value, unit=value.unit)

    def copy(self, name=None, **kwargs):
        """Copy SkyModel"""
        if self.spatial_model is not None:
            spatial_model = self.spatial_model.copy()
        else:
            spatial_model = None

        if self.temporal_model is not None:
            temporal_model = self.temporal_model.copy()
        else:
            temporal_model = None

        kwargs.setdefault("name", make_name(name))
        kwargs.setdefault("spectral_model", self.spectral_model.copy())
        kwargs.setdefault("spatial_model", spatial_model)
        kwargs.setdefault("temporal_model", temporal_model)
        kwargs.setdefault("apply_irf", self.apply_irf.copy())
        kwargs.setdefault("datasets_names", self.datasets_names)

        return self.__class__(**kwargs)

    def to_dict(self, full_output=False):
        """Create dict for YAML serilisation"""
        data = {}
        data["name"] = self.name
        data["type"] = self.tag[0]

        if self.apply_irf != self._apply_irf_default:
            data["apply_irf"] = self.apply_irf

        if self.datasets_names is not None:
            data["datasets_names"] = self.datasets_names

        data["spectral"] = self.spectral_model.to_dict(full_output)

        if self.spatial_model is not None:
            data["spatial"] = self.spatial_model.to_dict(full_output)

        if self.temporal_model is not None:
            data["temporal"] = self.temporal_model.to_dict(full_output)

        return data

    @classmethod
    def from_dict(cls, data):
        """Create SkyModel from dict"""
        from gammapy.modeling.models import (
            SPATIAL_MODEL_REGISTRY,
            SPECTRAL_MODEL_REGISTRY,
            TEMPORAL_MODEL_REGISTRY,
        )

        model_class = SPECTRAL_MODEL_REGISTRY.get_cls(data["spectral"]["type"])
        spectral_model = model_class.from_dict(data["spectral"])

        spatial_data = data.get("spatial")

        if spatial_data is not None:
            model_class = SPATIAL_MODEL_REGISTRY.get_cls(spatial_data["type"])
            spatial_model = model_class.from_dict(spatial_data)
        else:
            spatial_model = None

        temporal_data = data.get("temporal")

        if temporal_data is not None:
            model_class = TEMPORAL_MODEL_REGISTRY.get_cls(temporal_data["type"])
            temporal_model = model_class.from_dict(temporal_data)
        else:
            temporal_model = None

        return cls(
            name=data["name"],
            spatial_model=spatial_model,
            spectral_model=spectral_model,
            temporal_model=temporal_model,
            apply_irf=data.get("apply_irf", cls._apply_irf_default),
            datasets_names=data.get("datasets_names"),
        )

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n\n"

        str_ += "\t{:26}: {}\n".format("Name", self.name)

        str_ += "\t{:26}: {}\n".format("Datasets names", self.datasets_names)

        str_ += "\t{:26}: {}\n".format(
            "Spectral model type", self.spectral_model.__class__.__name__
        )

        if self.spatial_model is not None:
            spatial_type = self.spatial_model.__class__.__name__
        else:
            spatial_type = ""
        str_ += "\t{:26}: {}\n".format("Spatial  model type", spatial_type)

        if self.temporal_model is not None:
            temporal_type = self.temporal_model.__class__.__name__
        else:
            temporal_type = ""
        str_ += "\t{:26}: {}\n".format("Temporal model type", temporal_type)

        str_ += "\tParameters:\n"
        info = _get_parameters_str(self.parameters)
        lines = info.split("\n")
        str_ += "\t" + "\n\t".join(lines[:-1])

        str_ += "\n\n"
        return str_.expandtabs(tabsize=2)

    @classmethod
    def create(cls, spectral_model, spatial_model=None, temporal_model=None, **kwargs):
        """Create a model instance.

        Parameters
        ----------
        spectral_model : str
            Tag to create spectral model
        spatial_model : str
            Tag to create spatial model
        temporal_model : str
            Tag to create temporal model
        **kwargs : dict
            Keyword arguments passed to `SkyModel`

        Returns
        -------
        model : SkyModel
            Sky model
        """
        spectral_model = Model.create(spectral_model, model_type="spectral")

        if spatial_model:
            spatial_model = Model.create(spatial_model, model_type="spatial")

        if temporal_model:
            temporal_model = Model.create(temporal_model, model_type="temporal")

        return cls(
            spectral_model=spectral_model,
            spatial_model=spatial_model,
            temporal_model=temporal_model,
            **kwargs,
        )

    @classmethod
    def from_npred_template(cls, npred, spectral_model=None, name=None):
        """Create npred template.


        Parameters
        ----------
        npred : `Map`
            Npred template map.
        spectral_model : `NormSpectralModel`
            Norm spectral model
        name : str
            Name of the model.

        Returns
        -------
        model : `SkyModel`
            Npred template model

        """
        geom = npred.geom
        data = npred / geom.bin_volume()
        m = Map.from_geom(data=data.data, geom=geom.as_energy_true, unit=data.unit)

        spatial_model = TemplateSpatialModel(m, normalize=False)

        if spectral_model is None:
            spectral_model = PowerLawNormSpectralModel()

        return cls(
            spectral_model=spectral_model,
            spatial_model=spatial_model,
            apply_irf={"psf": False, "edisp": False, "exposure": False},
            name=name
        )


class FoVBackgroundModel(Model):
    """Field of view background model

    The background model holds the correction parameters applied to
    the instrumental background attached to a `MapDataset` or
    `SpectrumDataset`.

    Parameters
    ----------
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Normalized spectral model.
    dataset_name : str
        Dataset name

    """

    tag = ["FoVBackgroundModel", "fov-bkg"]

    def __init__(self, spectral_model=None, dataset_name=None):
        if dataset_name is None:
            raise ValueError("Dataset name a is required argument")

        self.datasets_names = [dataset_name]

        if spectral_model is None:
            spectral_model = PowerLawNormSpectralModel()

        if not spectral_model.is_norm_spectral_model:
            raise ValueError("A norm spectral model is required.")

        self._spectral_model = spectral_model
        super().__init__()

    @property
    def spectral_model(self):
        """Spectral norm model"""
        return self._spectral_model

    @property
    def name(self):
        """Model name"""
        return self.datasets_names[0] + "-bkg"

    @property
    def parameters(self):
        """Model parameters"""
        parameters = []
        parameters.append(self.spectral_model.parameters)
        return Parameters.from_stack(parameters)

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n\n"

        str_ += "\t{:26}: {}\n".format("Name", self.name)
        str_ += "\t{:26}: {}\n".format("Datasets names", self.datasets_names)
        str_ += "\t{:26}: {}\n".format(
            "Spectral model type", self.spectral_model.__class__.__name__
        )
        str_ += "\tParameters:\n"
        info = _get_parameters_str(self.parameters)
        lines = info.split("\n")
        str_ += "\t" + "\n\t".join(lines[:-1])

        str_ += "\n\n"
        return str_.expandtabs(tabsize=2)

    def evaluate_geom(self, geom):
        """Evaluate map"""
        energy = geom.axes["energy"].center[:, np.newaxis, np.newaxis]
        return self.evaluate(energy=energy)

    def evaluate(self, energy):
        """Evaluate model"""
        return self.spectral_model(energy)

    def to_dict(self, full_output=False):
        data = {}
        data["type"] = self.tag[0]
        data["datasets_names"] = self.datasets_names
        data["spectral"] = self.spectral_model.to_dict(full_output=full_output)
        return data

    @classmethod
    def from_dict(cls, data):
        """Create model from dict

        Parameters
        ----------
        data : dict
            Data dictionary
        """
        from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY

        spectral_data = data.get("spectral")
        if spectral_data is not None:
            model_class = SPECTRAL_MODEL_REGISTRY.get_cls(spectral_data["type"])
            spectral_model = model_class.from_dict(spectral_data)
        else:
            spectral_model = None

        datasets_names = data.get("datasets_names")

        if datasets_names is None:
            raise ValueError("FoVBackgroundModel must define a dataset name")

        if len(datasets_names) > 1:
            raise ValueError("FoVBackgroundModel can only be assigned to one dataset")

        return cls(spectral_model=spectral_model, dataset_name=datasets_names[0],)

    def reset_to_default(self):
        """Reset parameter values to default"""
        values = self.spectral_model.default_parameters.values
        self.spectral_model.parameters.values = values

    def copy(self, **kwargs):
        """Copy SkyModel"""
        return self.__class__(**kwargs)


def create_fermi_isotropic_diffuse_model(filename, **kwargs):
    """Read Fermi isotropic diffuse model.

    See `LAT Background models <https://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html>`_

    Parameters
    ----------
    filename : str
        filename
    kwargs : dict
        Keyword arguments forwarded to `TemplateSpectralModel`

    Returns
    -------
    diffuse_model : `SkyModel`
        Fermi isotropic diffuse sky model.
    """
    vals = np.loadtxt(make_path(filename))
    energy = u.Quantity(vals[:, 0], "MeV", copy=False)
    values = u.Quantity(vals[:, 1], "MeV-1 s-1 cm-2", copy=False)

    kwargs.setdefault("interp_kwargs", {"fill_value": None})

    spatial_model = ConstantSpatialModel()
    spectral_model = (
        TemplateSpectralModel(energy=energy, values=values, **kwargs)
        * PowerLawNormSpectralModel()
    )
    return SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        name="fermi-diffuse-iso",
        apply_irf={"psf": False, "exposure": True, "edisp": True},
    )
