# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from functools import lru_cache
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.nddata.utils import NoOverlapError
from astropy.utils import lazyproperty
from gammapy.irf import EnergyDependentMultiGaussPSF
from gammapy.maps import Map, WcsGeom
from gammapy.modeling.models import BackgroundModel
from .background import make_map_background_irf
from .edisp_map import make_edisp_map
from .exposure import _map_spectrum_weight, make_map_exposure_true_energy
from .fit import (
    BINSZ_IRF_DEFAULT,
    MARGIN_IRF_DEFAULT,
    MIGRA_AXIS_DEFAULT,
    RAD_AXIS_DEFAULT,
    MapDataset,
)
from .psf_map import make_psf_map

__all__ = ["MapDatasetMaker", "MapMakerRing"]

log = logging.getLogger(__name__)


class MapDatasetMaker:
    """Make maps for a single IACT observation.

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry in reco energy, used for counts and background maps
    offset_max : `~astropy.coordinates.Angle`
        Maximum offset angle
    energy_axis_true: `~gammapy.maps.MapAxis`
        True energy axis used for IRF maps
    migra_axis : `~gammapy.maps.MapAxis`
        Migration axis for edisp map
    rad_axis : `~gammapy.maps.MapAxis`
        Radial axis for psf map.
    binsz_irf: float
        IRF Map pixel size in degrees.
    margin_irf: float
        IRF map margin size in degrees
    cutout : bool
         Whether to cutout the observation.
    cutout_mode : {'trim', 'partial', 'strict'}
        Mode option for cutting out the observation,
        for details see `~astropy.nddata.utils.Cutout2D`.
    """

    def __init__(
        self,
        geom,
        offset_max,
        background_oversampling=None,
        energy_axis_true=None,
        migra_axis=None,
        rad_axis=None,
        binsz_irf=None,
        margin_irf=None,
        cutout_mode="trim",
        cutout=True,
    ):

        self.geom = geom
        self.offset_max = Angle(offset_max)
        self.background_oversampling = background_oversampling
        self.migra_axis = migra_axis if migra_axis else MIGRA_AXIS_DEFAULT
        self.rad_axis = rad_axis if rad_axis else RAD_AXIS_DEFAULT
        self.energy_axis_true = energy_axis_true or geom.get_axis_by_name("energy")
        self.binsz_irf = binsz_irf or BINSZ_IRF_DEFAULT

        self.margin_irf = margin_irf or MARGIN_IRF_DEFAULT
        self.margin_irf = self.margin_irf * u.deg

        self.cutout_mode = cutout_mode
        self.cutout_width = 2 * self.offset_max
        self.cutout = cutout

    def _cutout_geom(self, geom, observation):
        if self.cutout:
            return geom.cutout(
                position=observation.pointing_radec,
                width=self.cutout_width,
                mode=self.cutout_mode,
            )
        else:
            return geom

    @lazyproperty
    def geom_image_irf(self):
        """Spatial geometry of IRF Maps (`Geom`)"""
        geom_image = self.geom.to_image()

        if isinstance(self.geom, WcsGeom):
            geom_irf = WcsGeom.create(
                binsz=self.binsz_irf,
                width=geom_image.width + self.margin_irf,
                skydir=geom_image.center_skydir,
                proj=geom_image.projection,
                coordsys=geom_image.coordsys,
            )
        else:
            geom_irf = geom_image

        return geom_irf

    @lazyproperty
    def geom_exposure_irf(self):
        """Geom of Exposure map associated with IRFs (`Geom`)"""
        return self.geom_image_irf.to_cube([self.energy_axis_true])

    @lazyproperty
    def geom_exposure(self):
        """Exposure map geom (`Geom`)"""
        geom_exposure = self.geom.to_image().to_cube([self.energy_axis_true])
        return geom_exposure

    @lazyproperty
    def geom_psf(self):
        """PSFMap geom (`Geom`)"""
        geom_psf = self.geom_image_irf.to_cube([self.rad_axis, self.energy_axis_true])
        return geom_psf

    @lazyproperty
    def geom_edisp(self):
        """EdispMap geom (`Geom`)"""
        geom_edisp = self.geom_image_irf.to_cube(
            [self.migra_axis, self.energy_axis_true]
        )
        return geom_edisp

    def make_counts(self, observation):
        """Make counts map.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        counts : `Map`
            Counts map.
        """
        geom = self._cutout_geom(self.geom, observation)
        counts = Map.from_geom(geom)
        counts.fill_events(observation.events)
        return counts

    def make_exposure(self, observation):
        """Make exposure map.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        exposure : `Map`
            Exposure map.
        """
        geom = self._cutout_geom(self.geom_exposure, observation)
        exposure = make_map_exposure_true_energy(
            pointing=observation.pointing_radec,
            livetime=observation.observation_live_time_duration,
            aeff=observation.aeff,
            geom=geom,
        )
        return exposure

    @lru_cache(maxsize=1)
    def make_exposure_irf(self, observation):
        """Make exposure map with irf geometry.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        exposure : `Map`
            Exposure map.
        """

        geom = self._cutout_geom(self.geom_exposure_irf, observation)
        exposure = make_map_exposure_true_energy(
            pointing=observation.pointing_radec,
            livetime=observation.observation_live_time_duration,
            aeff=observation.aeff,
            geom=geom,
        )
        return exposure

    def make_background(self, observation):
        """Make background map.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        background : `Map`
            Background map.
        """
        geom = self._cutout_geom(self.geom, observation)

        bkg_coordsys = observation.bkg.meta.get("FOVALIGN", "ALTAZ")
        if bkg_coordsys == "ALTAZ":
            pointing = observation.fixed_pointing_info
        elif bkg_coordsys == "RADEC":
            pointing = observation.pointing_radec
        else:
            raise ValueError(
                f"Invalid background coordinate system: {bkg_coordsys!r}\n"
                "Options: ALTAZ, RADEC"
            )
        background = make_map_background_irf(
            pointing=pointing,
            ontime=observation.observation_time_duration,
            bkg=observation.bkg,
            geom=geom,
            oversampling=self.background_oversampling,
        )
        return background

    def make_edisp(self, observation):
        """Make edisp map.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        edisp : `EdispMap`
            Edisp map.
        """
        geom = self._cutout_geom(self.geom_edisp, observation)

        exposure = self.make_exposure_irf(observation)

        edisp = make_edisp_map(
            edisp=observation.edisp,
            pointing=observation.pointing_radec,
            geom=geom,
            max_offset=self.offset_max,
            exposure_map=exposure,
        )
        return edisp

    def make_psf(self, observation):
        """Make psf map.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        psf : `PSFMap`
            Psf map.
        """
        psf = observation.psf
        geom = self._cutout_geom(self.geom_psf, observation)

        if isinstance(psf, EnergyDependentMultiGaussPSF):
            psf = psf.to_psf3d(rad=self.rad_axis.center)

        exposure = self.make_exposure_irf(observation)

        psf = make_psf_map(
            psf=psf,
            pointing=observation.pointing_radec,
            geom=geom,
            max_offset=self.offset_max,
            exposure_map=exposure,
        )
        return psf

    @lru_cache(maxsize=1)
    def make_mask_safe(self, observation):
        """Make offset mask.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        mask : `Map`
            Mask
        """
        geom = self._cutout_geom(self.geom.to_image(), observation)
        offset = geom.separation(observation.pointing_radec)
        data = offset >= self.offset_max
        return Map.from_geom(geom, data=data)

    @lru_cache(maxsize=1)
    def make_mask_safe_irf(self, observation):
        """Make offset mask with irf geometry.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        mask : `Map`
            Mask
        """
        geom = self._cutout_geom(self.geom_exposure_irf.to_image(), observation)
        offset = geom.separation(observation.pointing_radec)
        data = offset >= self.offset_max
        return Map.from_geom(geom, data=data)

    def run(self, observation, selection=None):
        """Make map dataset.

        Parameters
        ----------
        observation : `~gammapy.data.DataStoreObservation`
            Observation
        selection : list
            List of str, selecting which maps to make.
            Available: 'counts', 'exposure', 'background', 'psf', 'edisp'
            By default, all maps are made.

        Returns
        -------
        dataset : `MapDataset`
            Map dataset.

        """
        selection = _check_selection(selection)

        mask_safe = self.make_mask_safe(observation)
        energy_axis = self.geom.get_axis_by_name("energy")
        mask_safe_3d = (
            ~mask_safe.data
            & np.ones(energy_axis.nbin, dtype=bool)[:, np.newaxis, np.newaxis]
        )
        mask_map = Map.from_geom(
            mask_safe.geom.to_cube([energy_axis]), data=mask_safe_3d
        )
        mask_safe_irf = self.make_mask_safe_irf(observation)

        kwargs = {
            "name": f"obs_{observation.obs_id}",
            "gti": observation.gti,
            "mask_safe": mask_map,
        }

        if "counts" in selection:
            counts = self.make_counts(observation)
            # TODO: remove masking out the values here and instead handle the safe mask only when
            #  fitting and / or stacking datasets?
            counts.data[..., mask_safe.data] = 0
            kwargs["counts"] = counts

        if "exposure" in selection:
            exposure = self.make_exposure(observation)
            exposure.data[..., mask_safe.data] = 0
            kwargs["exposure"] = exposure

        if "background" in selection:
            background_map = self.make_background(observation)
            background_map.data[..., mask_safe.data] = 0
            kwargs["background_model"] = BackgroundModel(background_map)

        if "psf" in selection:
            psf = self.make_psf(observation)
            psf.exposure_map.data[..., mask_safe_irf.data] = 0
            kwargs["psf"] = psf

        if "edisp" in selection:
            edisp = self.make_edisp(observation)
            psf.exposure_map.data[..., mask_safe_irf.data] = 0
            kwargs["edisp"] = edisp

        return MapDataset(**kwargs)


def _check_selection(selection):
    """Handle default and validation of selection"""
    available = ["counts", "exposure", "background", "psf", "edisp"]
    if selection is None:
        selection = available

    if not isinstance(selection, list):
        raise TypeError("Selection must be a list of str")

    for name in selection:
        if name not in available:
            raise ValueError(f"Selection not available: {name!r}")

    return selection


class MapMakerRing:
    """Make maps from IACT observations.

    The main motivation for this class in addition to the `MapMaker`
    is to have the common image background estimation methods,
    like `~gammapy.cube.RingBackgroundEstimator`,
    that work using on and off maps.

    To ensure adequate statistics, only observations that are fully
    contained within the reference geometry will be analysed

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry
    offset_max : `~astropy.coordinates.Angle`
        Maximum offset angle
    exclusion_mask : `~gammapy.maps.Map`
        Exclusion mask
    background_estimator : `~gammapy.cube.RingBackgroundEstimator`
        or `~gammapy.cube.AdaptiveRingBackgroundEstimator`
        Ring background estimator or something with an equivalent API.

    Examples
    --------
    Here is an example how to ise the MapMakerRing with H.E.S.S. DL3 data::

        import numpy as np
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from regions import CircleSkyRegion
        from gammapy.maps import Map, WcsGeom, MapAxis
        from gammapy.cube import MapMakerRing, RingBackgroundEstimator
        from gammapy.data import DataStore

        # Create observation list
        data_store = DataStore.from_dir(
            "$GAMMAPY_DATA/hess-dl3-dr1/"
        )
        data_sel = data_store.obs_table["TARGET_NAME"] == "MSH 15-52"
        obs_table = data_store.obs_table[data_sel]
        observations = data_store.get_observations(obs_table["OBS_ID"])

        # Define the geom
        pos = SkyCoord(228.32, -59.08, unit="deg")
        energy_axis = MapAxis.from_edges(np.logspace(0, 5.0, 5), unit="TeV", name="energy")
        geom = WcsGeom.create(skydir=pos, binsz=0.02, width=(5, 5), axes=[energy_axis])

        # Make a region mask
        regions = CircleSkyRegion(center=pos, radius=0.3 * u.deg)
        mask = Map.from_geom(geom)
        mask.data = mask.geom.region_mask([regions], inside=False)

        # Run map maker with ring background estimation
        ring_bkg = RingBackgroundEstimator(r_in="0.5 deg", width="0.3 deg")
        maker = MapMakerRing(
            geom=geom, offset_max="2 deg", exclusion_mask=mask, background_estimator=ring_bkg
        )
        images = maker.run_images(observations)
    """

    def __init__(
        self, geom, offset_max, exclusion_mask=None, background_estimator=None
    ):

        self.geom = geom
        self.offset_max = Angle(offset_max)
        self.exclusion_mask = exclusion_mask
        self.background_estimator = background_estimator

    def _get_empty_maps(self, selection):
        # Initialise zero-filled maps
        maps = {}
        for name in selection:
            if name == "exposure":
                maps[name] = Map.from_geom(self.geom, unit="m2 s")
            else:
                maps[name] = Map.from_geom(self.geom, unit="")
        return maps

    def _get_obs_maker(self, obs):
        # Compute cutout geometry and slices to stack results back later

        # Make maps for this observation
        return MapDatasetMaker(geom=self.geom, offset_max=self.offset_max)

    @staticmethod
    def _maps_sum_over_axes(maps, spectrum, keepdims):
        """Compute weighted sum over map axes.

        Parameters
        ----------
        spectrum : `~gammapy.modeling.models.SpectralModel`
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.
        keepdims : bool, optional
            If this is set to True, the energy axes is kept with a single bin.
            If False, the energy axes is removed
        """
        images = {}
        for name, map in maps.items():
            if name == "exposure":
                map = _map_spectrum_weight(map, spectrum)
            images[name] = map.sum_over_axes(keepdims=keepdims)
        # TODO: PSF (and edisp) map sum_over_axis

        return images

    def _run(self, observations, sum_over_axis=False, spectrum=None, keepdims=False):
        selection = ["on", "exposure_on", "off", "exposure_off", "exposure"]
        maps = self._get_empty_maps(selection)
        if sum_over_axis:
            maps = self._maps_sum_over_axes(maps, spectrum, keepdims)

        for obs in observations:
            try:
                obs_maker = self._get_obs_maker(obs)
            except NoOverlapError:
                log.info(f"Skipping obs_id: {obs.obs_id} (no map overlap)")
                continue

            dataset = obs_maker.run(obs, selection=["counts", "exposure", "background"])
            maps_obs = {}
            maps_obs["counts"] = dataset.counts
            maps_obs["exposure"] = dataset.exposure
            maps_obs["background"] = dataset.background_model.map
            maps_obs["exclusion"] = self.exclusion_mask.cutout(
                position=obs.pointing_radec, width=2 * self.offset_max, mode="trim"
            )

            if sum_over_axis:
                maps_obs = self._maps_sum_over_axes(maps_obs, spectrum, keepdims)
                maps_obs["exclusion"] = maps_obs["exclusion"].sum_over_axes(
                    keepdims=keepdims
                )
                maps_obs["exclusion"].data = (
                    maps_obs["exclusion"].data / self.geom.axes[0].nbin
                )

            maps_obs_bkg = self.background_estimator.run(maps_obs)
            maps_obs.update(maps_obs_bkg)
            maps_obs["exposure_on"] = maps_obs.pop("background")
            maps_obs["on"] = maps_obs.pop("counts")

            # Now paste the returned maps on the ref geom
            for name in selection:
                data = maps_obs[name].quantity.to_value(maps[name].unit)
                maps[name].fill_by_coord(maps_obs[name].geom.get_coord(), data)

        self._maps = maps
        return maps

    def run_images(self, observations, spectrum=None, keepdims=False):
        """Run image making.

        The maps are summed over on the energy axis for a classical image analysis.

        Parameters
        ----------
        observations : `~gammapy.data.Observations`
            Observations to process
        spectrum : `~gammapy.modeling.models.SpectralModel`, optional
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.
        keepdims : bool, optional
            If this is set to True, the energy axes is kept with a single bin.
            If False, the energy axes is removed

        Returns
        -------
        maps : dict of `~gammapy.maps.Map`
            Dictionary containing the following maps:

            * ``"on"``: counts map
            * ``"exposure_on"``: on exposure map, which is just the
              template background map from the IRF
            * ``"exposure_off"``: off exposure map convolved with the ring
            * ``"off"``: off map
        """
        return self._run(
            observations, sum_over_axis=True, spectrum=spectrum, keepdims=keepdims
        )

    def run(self, observations):
        """Run map making.

        Parameters
        ----------
        observations : `~gammapy.data.Observations`
            Observations to process

        Returns
        -------
        maps : dict of `~gammapy.maps.Map`
            Dictionary containing the following maps:

            * ``"on"``: counts map
            * ``"exposure_on"``: on exposure map, which is just the
              template background map from the IRF
            * ``"exposure_off"``: off exposure map convolved with the ring
            * ``"off"``: off map
        """
        return self._run(observations, sum_over_axis=False)
