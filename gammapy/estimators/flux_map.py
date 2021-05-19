# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.utils import classproperty
from gammapy.data import GTI
from gammapy.maps import MapCoord, Map
from gammapy.estimators.core import FluxEstimate, OPTIONAL_QUANTITIES_COMMON, REQUIRED_MAPS, OPTIONAL_QUANTITIES
from gammapy.estimators.flux_point import FluxPoints
from gammapy.modeling.models import SkyModel, Models, PowerLawSpectralModel
from gammapy.utils.scripts import make_path

__all__ = ["FluxMaps"]


log = logging.getLogger(__name__)


class FluxMaps(FluxEstimate):
    """A flux map container.

    It contains a set of `~gammapy.maps.Map` objects that store the estimated
    flux as a function of energy as well asn associated quantities (typically
    errors, upper limits, delta TS and possibly raw quantities such counts,
    excesses etc). It also contains a reference model to convert the flux values
    in different formats. Usually, this should be the model used to produce the
    flux map.

    The associated map geometry can use a `RegionGeom` to store the equivalent
    of flux points, or a `WcsGeom`/`HpxGeom` to store an energy dependent flux
    map.

    The container relies internally on the 'Likelihood' SED type defined in
    :ref:`gadf:flux-points` and offers convenience properties to convert to
    other flux formats, namely: ``dnde``, ``flux``, ``eflux`` or ``e2dnde``.
    The conversion is done according to the reference model spectral shape.

    Parameters
    ----------
    data : dict of `~gammapy.maps.Map`
        the maps dictionary. Expected entries are the following:
        * norm : the norm factor
        * norm_err : optional, the error on the norm factor.
        * norm_errn : optional, the negative error on the norm factor.
        * norm_errp : optional, the positive error on the norm factor.
        * norm_ul : optional, the upper limit on the norm factor.
        * norm_scan : optional, the norm values of the test statistic scan.
        * stat_scan : optional, the test statistic scan values.
        * ts : optional, the delta TS associated with the flux value.
        * sqrt_ts : optional, the square root of the TS, when relevant.
    gti : `~gammapy.data.GTI`
        the maps GTI information. Default is None.
    """

    def __init__(self, data, gti=None):
        self.gti = gti
        super().__init__(data=data)

    @classproperty
    def reference_model_default(cls):
        """Default reference model: a point source with index = 2  (`SkyModel`)"""
        return SkyModel.create("pl", "point")

    @property
    def geom(self):
        """Reference map geometry (`Geom`)"""
        return self.data["norm"].geom

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__)
        str_ += "\n\n"
        str_ += "\t" + "\t\n".join(str(self.geom).split("\n")[:1])
        str_ += "\n\t" + "\n\t".join(str(self.geom).split("\n")[2:])

        str_ += f"\n\tAvailable quantities : {list(self.data.keys())}\n\n"

        return str_.expandtabs(tabsize=2)

    def get_flux_points(self, position=None):
        """Extract flux point at a given position.

        Parameters
        ---------
        position : `~astropy.coordinates.SkyCoord`
            Position where the flux points are extracted.

        Returns
        -------
        flux_points : `~gammapy.estimators.FluxPoints`
            Flux points object
        """
        if position is None:
            position = self.geom.center_skydir

        table = Table()
        table.meta["SED_TYPE"] = "likelihood"

        coords = MapCoord.create(
            {"skycoord": position, "energy": self.energy_ref}
        )
        table["e_min"] = self.energy_min
        table["e_max"] = self.energy_max
        table["e_ref"] = self.energy_ref

        # TODO: add support of norm and stat scan
        for name in self.data:
            m = getattr(self, name)
            table[name] = m.get_by_coord(coords) * m.unit

        return FluxPoints(data=table)

    def to_dict(self, sed_type="likelihood"):
        """Return maps in a given SED type in the form of a dictionary.

        Parameters
        ----------
        sed_type : str
            sed type to convert to. Default is `Likelihood`

        Returns
        -------
        map_dict : dict
            Dictionary containing the requested maps.
        """
        if sed_type == "likelihood":
            data = self.data
        else:
            data = {}
            all_maps = REQUIRED_MAPS[sed_type] + OPTIONAL_QUANTITIES[sed_type] + OPTIONAL_QUANTITIES_COMMON

            for quantity in all_maps:
                try:
                    data[quantity] = getattr(self, quantity)
                except KeyError:
                    pass

        return data

    def write(self, filename, overwrite=False, sed_type="likelihood"):
        """Write flux map to file.

        Parameters
        ----------
        filename : str
            Filename to write to.
        overwrite : bool
            Overwrite file if it exists.
        sed_type : str
            sed type to convert to. Default is `likelihood`
        """
        filename = make_path(filename)
        hdulist = self.to_hdulist(sed_type)
        hdulist.writeto(filename, overwrite=overwrite)

    def to_hdulist(self, sed_type="likelihood", hdu_bands=None):
        """Convert flux map to list of HDUs.

        For now, one cannot export the reference model.

        Parameters
        ----------
        sed_type : str
            sed type to convert to. Default is `Likelihood`
        hdu_bands : str
            Name of the HDU with the BANDS table. Default is 'BANDS'
            If set to None, each map will have its own hdu_band

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
            Map dataset list of HDUs.
        """
        exclude_primary = slice(1, None)

        hdu_primary = fits.PrimaryHDU()
        hdu_primary.header["SED_TYPE"] = sed_type
        hdulist = fits.HDUList([hdu_primary])

        data = self.to_dict(sed_type)

        for key, m in data.items():
            hdulist += m.to_hdulist(hdu=key, hdu_bands=hdu_bands)[
                exclude_primary
            ]

        if self.gti:
            hdu = fits.BinTableHDU(self.gti.table, name="GTI")
            hdulist.append(hdu)

        return hdulist

    @classmethod
    def read(cls, filename):
        """Read map dataset from file.

        Parameters
        ----------
        filename : str
            Filename to read from.

        Returns
        -------
        flux_maps : `~gammapy.estimators.FluxMaps`
            Flux maps object.
        """
        with fits.open(str(make_path(filename)), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist)

    @classmethod
    def from_hdulist(cls, hdulist, hdu_bands=None):
        """Create flux map dataset from list of HDUs.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            List of HDUs.
        hdu_bands : str
            Name of the HDU with the BANDS table. Default is 'BANDS'
            If set to None, each map should have its own hdu_band

        Returns
        -------
        flux_maps : `~gammapy.estimators.FluxMaps`
            Flux maps object.
        """
        try:
            sed_type = hdulist[0].header["SED_TYPE"]
        except KeyError:
            raise ValueError(
                f"Cannot determine SED type of flux map from primary header."
            )

        maps = {}

        for map_type in REQUIRED_MAPS[sed_type]:
            maps[map_type] = Map.from_hdulist(
                hdulist, hdu=map_type, hdu_bands=hdu_bands
            )

        for map_type in OPTIONAL_QUANTITIES[sed_type] + OPTIONAL_QUANTITIES_COMMON:
            if map_type.upper() in hdulist:
                maps[map_type] = Map.from_hdulist(
                    hdulist, hdu=map_type, hdu_bands=hdu_bands
                )

        if "GTI" in hdulist:
            gti = GTI(Table.read(hdulist["GTI"]))
        else:
            gti = None

        return cls.from_dict(maps=maps, sed_type=sed_type, gti=gti)

    @classmethod
    def from_dict(cls, maps, sed_type="likelihood", reference_model=None, gti=None):
        """Create FluxMaps from a dictionary of maps.

        Parameters
        ----------
        maps : dict
            Dictionary containing the input maps.
        sed_type : str
            SED type of the input maps. Default is `Likelihood`
        reference_model : `~gammapy.modeling.models.SkyModel`, optional
            Reference model to use for conversions. Default in None.
            If None, a model consisting of a point source with a power law
            spectrum of index 2 is assumed.
        gti : `~gammapy.data.GTI`
            Maps GTI information. Default is None.

        Returns
        -------
        flux_maps : `~gammapy.estimators.FluxMaps`
            Flux maps object.
        """
        cls._validate_data(data=maps, sed_type=sed_type)

        if sed_type == "likelihood":
            return cls(data=maps, gti=gti)

        if reference_model is None:
            log.warning(
                "No reference model set for FluxMaps. Assuming point source with E^-2 spectrum."
            )
            reference_model = PowerLawSpectralModel()

        map_ref = maps[sed_type]

        with np.errstate(invalid="ignore", divide="ignore"):
            fluxes = reference_model.to_reference_flux_maps(geom=map_ref.geom)

        factor = fluxes[f"ref_{sed_type}"]

        data = dict()
        data["norm"] = (map_ref / factor).to_unit("")

        for key in OPTIONAL_QUANTITIES[sed_type]:
            if key in maps:
                norm_type = key.replace(sed_type, "norm")
                data[norm_type] = (maps[key] / factor).to_unit("")

        # We add the remaining maps
        for key in OPTIONAL_QUANTITIES_COMMON:
            if key in maps:
                data[key] = maps[key]

        return cls(data=data, gti=gti)

    # TODO: should we allow this?
    def __getitem__(self, item):
        return getattr(self, item)
