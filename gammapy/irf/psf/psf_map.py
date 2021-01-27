# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from gammapy.maps import Map, MapCoord, WcsGeom, MapAxes
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.utils.random import InverseCDFSampler, get_random_state
from gammapy.utils.gauss import Gauss2DPDF
from .kernel import PSFKernel
from .table import EnergyDependentTablePSF
from ..core import IRFMap

__all__ = ["PSFMap"]


class PSFMap(IRFMap):
    """Class containing the Map of PSFs and allowing to interact with it.

    Parameters
    ----------
    psf_map : `~gammapy.maps.Map`
        the input PSF Map. Should be a Map with 2 non spatial axes.
        rad and true energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`
        Associated exposure map. Needs to have a consistent map geometry.

    Examples
    --------
    ::

        import numpy as np
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from gammapy.maps import Map, WcsGeom, MapAxis
        from gammapy.irf import EnergyDependentMultiGaussPSF, EffectiveAreaTable2D
        from gammapy.makers.utils import make_psf_map, PSFMap, make_map_exposure_true_energy

        # Define energy axis. Note that the name is fixed.
        energy_axis = MapAxis.from_edges(np.logspace(-1., 1., 4), unit='TeV', name='energy')
        # Define rad axis. Again note the axis name
        rads = np.linspace(0., 0.5, 100) * u.deg
        rad_axis = MapAxis.from_edges(rads, unit='deg', name='theta')

        # Define parameters
        pointing = SkyCoord(0., 0., unit='deg')
        max_offset = 4 * u.deg

        # Create WcsGeom
        geom = WcsGeom.create(binsz=0.25*u.deg, width=10*u.deg, skydir=pointing, axes=[rad_axis, energy_axis])

        # Extract EnergyDependentTablePSF from CTA 1DC IRF
        filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
        psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
        psf3d = psf.to_psf3d(rads)
        aeff2d = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')

        # Create the exposure map
        exposure_geom = geom.to_image().to_cube([energy_axis])
        exposure_map = make_map_exposure_true_energy(pointing, "1 h", aeff2d, exposure_geom)

        # create the PSFMap for the specified pointing
        psf_map = make_psf_map(psf3d, pointing, geom, max_offset, exposure_map)

        # Get an EnergyDependentTablePSF at any position in the image
        psf_table = psf_map.get_energy_dependent_table_psf(SkyCoord(2., 2.5, unit='deg'))

        # Write map to disk
        psf_map.write('psf_map.fits')
    """
    tag = "psf_map"
    required_axes = ["rad", "energy_true"]

    def __init__(self, psf_map, exposure_map=None):
        super().__init__(irf_map=psf_map, exposure_map=exposure_map)

    @property
    def psf_map(self):
        return self._irf_map

    @psf_map.setter
    def psf_map(self, value):
        self._irf_map = value

    def to_region_nd_map(self, region):
        """Convert to region ND PSF map

        If a region is given a mean PSF is computed.

        Parameters
        ----------
        region : `SkyRegion` or `SkyCoord`
            Region or position where to get the map.

        Returns
        -------
        psf : `PSFMap`
            PSF map with region geometry.
        """
        if region is None:
            region = self.psf_map.geom.center_skydir

        # TODO: compute an exposure weighted mean PSF here
        kwargs = {"region": region, "func": np.nanmean}
        psf_map = self.psf_map.to_region_nd_map(**kwargs)

        if self.exposure_map:
            exposure_map = self.exposure_map.to_region_nd_map(**kwargs)
        else:
            exposure_map = None

        return self.__class__(
            psf_map=psf_map,
            exposure_map=exposure_map
        )

    def get_energy_dependent_table_psf(self, position=None):
        """Get energy-dependent PSF at a given position.

        By default the PSF at the center of the map is returned.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            the target position. Should be a single coordinates

        Returns
        -------
        psf_table : `~gammapy.irf.EnergyDependentTablePSF`
            the table PSF
        """
        psf_map = self.to_region_nd_map(region=position)

        data = psf_map.psf_map.quantity.squeeze()

        if self.exposure_map:
            exposure = psf_map.exposure_map.quantity.squeeze()
        else:
            exposure = None

        # Beware. Need to revert rad and energies to follow the TablePSF scheme.
        return EnergyDependentTablePSF(
            axes=self.psf_map.geom.axes[["energy_true", "rad"]],
            data=data.value,
            unit=data.unit,
            exposure=exposure,
        )

    def get_psf_kernel(self, position, geom, max_radius=None, factor=4):
        """Returns a PSF kernel at the given position.

        The PSF is returned in the form a WcsNDMap defined by the input Geom.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            the target position. Should be a single coordinate
        geom : `~gammapy.maps.Geom`
            the target geometry to use
        max_radius : `~astropy.coordinates.Angle`
            maximum angular size of the kernel map
        factor : int
            oversampling factor to compute the PSF

        Returns
        -------
        kernel : `~gammapy.irf.PSFKernel`
            the resulting kernel
        """
        if position is None:
            position = self.psf_map.geom.center_skydir

        table_psf = self.get_energy_dependent_table_psf(position)

        if max_radius is None:
            max_radius = np.max(table_psf.axes["rad"].center)
            min_radius_geom = np.min(geom.width) / 2.0
            max_radius = min(max_radius, min_radius_geom)

        return PSFKernel.from_table_psf(table_psf, geom, max_radius, factor)

    def containment_radius_map(self, energy, fraction=0.68):
        """Containment radius map.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Scalar energy at which to compute the containment radius
        fraction : float
            the containment fraction (range: 0 to 1)

        Returns
        -------
        containment_radius_map : `~gammapy.maps.Map`
            Containment radius map
        """
        coords = self.psf_map.geom.to_image().get_coord().skycoord.flatten()
        m = Map.from_geom(self.psf_map.geom.to_image(), unit="deg")

        for coord in coords:
            psf_table = self.get_energy_dependent_table_psf(coord)
            containment_radius = psf_table.containment_radius(
                energy_true=energy, fraction=fraction
            )
            m.fill_by_coord(coord, np.atleast_1d(containment_radius))

        return m

    @classmethod
    def from_geom(cls, geom):
        """Create psf map from geom.

        Parameters
        ----------
        geom : `Geom`
            PSF map geometry.

        Returns
        -------
        psf_map : `PSFMap`
            Point spread function map.
        """
        geom_exposure_psf = geom.squash(axis_name="rad")
        exposure_psf = Map.from_geom(geom_exposure_psf, unit="m2 s")
        psf_map = Map.from_geom(geom, unit="sr-1")
        return cls(psf_map, exposure_psf)

    def sample_coord(self, map_coord, random_state=0):
        """Apply PSF corrections on the coordinates of a set of simulated events.

        Parameters
        ----------
        map_coord : `~gammapy.maps.MapCoord` object.
            Sequence of coordinates and energies of sampled events.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.

        Returns
        -------
        corr_coord : `~gammapy.maps.MapCoord` object.
            Sequence of PSF-corrected coordinates of the input map_coord map.
        """

        random_state = get_random_state(random_state)
        rad_axis = self.psf_map.geom.axes["rad"]

        coord = {
            "skycoord": map_coord.skycoord.reshape(-1, 1),
            "energy_true": map_coord["energy_true"].reshape(-1, 1),
            "rad": rad_axis.center,
        }

        pdf = (
            self.psf_map.interp_by_coord(coord)
            * rad_axis.center.value
            * rad_axis.bin_width.value
        )

        sample_pdf = InverseCDFSampler(pdf, axis=1, random_state=random_state)
        pix_coord = sample_pdf.sample_axis()
        separation = rad_axis.pix_to_coord(pix_coord)

        position_angle = random_state.uniform(360, size=len(map_coord.lon)) * u.deg

        event_positions = map_coord.skycoord.directional_offset_by(
            position_angle=position_angle, separation=separation
        )
        return MapCoord.create(
            {"skycoord": event_positions, "energy_true": map_coord["energy_true"]}
        )

    @classmethod
    def from_energy_dependent_table_psf(cls, table_psf, geom=None):
        """Create PSF map from table PSF object.

        Helper function to create an allsky PSF map from
        table PSF, which does not depend on position.

        Parameters
        ----------
        table_psf : `EnergyDependentTablePSF`
            Table PSF

        Returns
        -------
        psf_map : `PSFMap`
            Point spread function map.
        """
        if geom is None:
            geom = WcsGeom.create(
                npix=(2, 1),
                proj="CAR",
                binsz=180,
                axes=table_psf.axes.reverse,
            )
        coords = geom.get_coord()

        data = table_psf.evaluate(
            energy_true=coords["energy_true"], rad=coords["rad"]
        )
        psf_map = Map.from_geom(geom, data=data.to_value("sr-1"), unit="sr-1")

        geom_exposure = geom.squash(axis_name="rad")

        exposure_map = Map.from_geom(geom_exposure, unit="cm2 s", data=1)
        data = table_psf.exposure
        exposure_map.quantity = exposure_map.data * data.reshape((-1, 1, 1, 1))
        return cls(psf_map=psf_map, exposure_map=exposure_map)

    @classmethod
    def from_gauss(cls, energy_axis_true, rad_axis=None, sigma=0.1 * u.deg):
        """Create all -sky PSF map from Gaussian width.

        This is used for testing and examples.

        The width can be the same for all energies
        or be an array with one value per energy node.
        It does not depend on position.

        Parameters
        ----------
        energy_axis_true : `~gammapy.maps.MapAxis`
            True energy axis.
        rad_axis : `~gammapy.maps.MapAxis`
            Offset angle wrt source position axis.
        sigma : `~astropy.coordinates.Angle`
            Gaussian width.

        Returns
        -------
        psf_map : `PSFMap`
            Point spread function map.
        """
        from gammapy.datasets.map import RAD_AXIS_DEFAULT

        if rad_axis is None:
            rad_axis = RAD_AXIS_DEFAULT.copy()

        axes = MapAxes([energy_axis_true, rad_axis])
        coords = axes.get_coord()

        sigma = np.broadcast_to(u.Quantity(sigma), energy_axis_true.nbin, subok=True)
        gauss = Gauss2DPDF(sigma=sigma.reshape((-1, 1)))
        data = gauss(coords["rad"])

        table_psf = EnergyDependentTablePSF(
            axes=axes, unit=data.unit, data=data.value
        )
        return cls.from_energy_dependent_table_psf(table_psf)

    def to_image(self, spectrum=None, keepdims=True):
        """Reduce to a 2-D map after weighing
        with the associated exposure and a spectrum

        Parameters
        ----------
        spectrum : `~gammapy.modeling.models.SpectralModel`, optional
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.
        keepdims : bool, optional
            If True, the energy axis is kept with one bin.
            If False, the axis is removed


        Returns
        -------
        psf_out : `PSFMap`
            `PSFMap` with the energy axis summed over
        """
        from gammapy.makers.utils import _map_spectrum_weight

        if spectrum is None:
            spectrum = PowerLawSpectralModel(index=2.0)

        exp_weighed = _map_spectrum_weight(self.exposure_map, spectrum)
        exposure = exp_weighed.sum_over_axes(
            axes_names=["energy_true"], keepdims=keepdims
        )

        psf_data = exp_weighed.data * self.psf_map.data / exposure.data
        psf_map = Map.from_geom(geom=self.psf_map.geom, data=psf_data, unit="sr-1")

        psf = psf_map.sum_over_axes(axes_names=["energy_true"], keepdims=keepdims)
        return self.__class__(psf_map=psf, exposure_map=exposure)
