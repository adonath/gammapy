import numpy as np
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_area
from regions import CircleSkyRegion
from gammapy.utils.regions import make_region
from .geom import Geom, axes_pix_to_coord, pix_tuple_to_idx, make_axes, frame_to_coordsys
from .wcs import WcsGeom
from .base import MapCoord


class RegionGeom(Geom):
    """Map geometry representing a region on the sky.

    Parameters
    ----------
    region : `~regions.SkyRegion`
        Region object.
    axes : list of `MapAxis`
        Non-spatial data axes.
    wcs : `~astropy.wcs.WCS`
        Optional wcs object to project the region if needed.
    """
    is_image = False
    is_allsky = False
    is_hpx = False
    _slice_spatial_axes = slice(0, 2)
    _slice_non_spatial_axes = slice(2, None)
    projection = "TAN"

    def __init__(self, region, axes=None, wcs=None):
        self._region = region
        self._axes = make_axes(axes)

        if wcs is None:
            wcs = WcsGeom.create(
                skydir=region.center, binsz=0.001, width=region.radius, proj="TAN"
            ).wcs

        self._wcs = wcs
        self.ndim = len(self.data_shape)
        self.coordsys = frame_to_coordsys(region.center.frame.name)

    @property
    def width(self):
        if isinstance(self.region, CircleSkyRegion):
            return 2 * self.region.radius
        else:
            raise ValueError("Only circular regions supported")

    @property
    def region(self):
        return self._region

    @property
    def axes(self):
        return self._axes

    @property
    def wcs(self):
        return self._wcs

    @property
    def center_coord(self):
        """(`astropy.coordinates.SkyCoord`)"""
        return self.pix_to_coord(self.center_pix)

    @property
    def center_pix(self):
        return tuple((np.array(self.data_shape) - 1.0) / 2)[::-1]

    @property
    def center_skydir(self):
        """Center skydir"""
        return self.region.center

    def contains(self, position):
        idx = self.coord_to_idx(coords)
        return np.all(np.stack([t != INVALID_INDEX.int for t in idx]), axis=0)

    def separation(self, position):
        coord = self.get_coord()
        return coord.skycoord.separation(position)

    @property
    def data_shape(self):
        return tuple([ax.nbin for ax in self.axes]) + (1, 1)

    def get_coord(self, coordsys=None):
        """Get map coordinates from the geometry.

        Returns
        -------
        coord : `~MapCoord`
            Map coordinate object.
        """
        cdict = {}
        cdict["skycoord"] = self.center_skydir.reshape((1, 1))

        if self.axes is not None:
            for ax in self.axes:
                cdict[ax.name] = ax.center.reshape((-1, 1, 1))

        if coordsys is None:
            coordsys = self.coordsys

        return MapCoord.create(cdict, coordsys=self.coordsys).to_coordsys(coordsys)

    def pad(self):
        raise NotImplementedError("Padding of `RegionGeom` not implemented")

    def crop(self):
        raise NotImplementedError("Cropping of `RegionGeom` not implemented")

    def solid_angle(self):
        area = self.region.to_pixel(self.wcs).area
        solid_angle = area * proj_plane_pixel_area(self.wcs) * u.deg ** 2
        return solid_angle.to("sr")

    def bin_volume(self):
        return self.solid_angle() * self.axes[0].bin_width.reshape((-1, 1, 1))

    def to_cube(self, axes):
        return self._init_copy(axes=axes)

    def to_image(self):
        return self._init_copy(axes=None)

    def upsample(self, factor, axis):
        axes = copy.deepcopy(self.axes)
        idx = self.get_axis_index_by_name(axis)
        axes[idx] = axes[idx].upsample(factor)
        return self._init_copy(axes=axes)

    def downsample(self, factor, axis):
        axes = copy.deepcopy(self.axes)
        idx = self.get_axis_index_by_name(axis)
        axes[idx] = axes[idx].downsample(factor)
        return self._init_copy(axes=axes)

    def pix_to_coord(self, pix):
        lon = np.where((-0.5 < pix[0]) & (pix[0] < 0.5), self.center_skydir.l.deg, np.nan * u.deg) * u.deg
        lat = np.where((-0.5 < pix[1]) & (pix[1] < 0.5), self.center_skydir.b.deg, np.nan * u.deg) * u.deg
        coords = (lon, lat)
        coords += tuple(axes_pix_to_coord(self.axes, pix[self._slice_non_spatial_axes]))
        return coords

    def pix_to_idx(self, pix, clip=True):
        idxs = list(pix_tuple_to_idx(pix))
        if True:
            idxs[0] = np.where((-0.5 < pix[0]) & (pix[0] < 0.5), [0], [-1])
            idxs[1] = np.where((-0.5 < pix[1]) & (pix[1] < 0.5), [0], [-1])
            idxs[2] = np.clip(idxs[2], 0, self.axes[0].nbin - 1)
        return tuple(idxs)

    def coord_to_pix(self, coords):
        coords = MapCoord.create(coords, coordsys=self.coordsys)
        in_region = self.region.contains(coords.skycoord, wcs=self.wcs)

        x = np.zeros(coords.shape)
        x[~in_region] = np.nan

        y = np.zeros(coords.shape)
        y[~in_region] = np.nan

        pix = (x, y)
        for coord, ax in zip(coords[self._slice_non_spatial_axes], self.axes):
            pix += (ax.coord_to_pix(coord),)

        return pix

    def get_idx(self):
        idxs = (0, 0)
        for ax in self.axes:
            idxs += np.arange(ax.nbin)
        return idxs

    def _make_bands_cols(self):
        pass

    @classmethod
    def create(cls, region, **kwargs):
        """Create region.

        Parameters
        ----------
        region : str or `~regions.SkyRegion`
            Region

        """
        if isinstance(region, str):
            region = make_region(region)

        return cls(region, **kwargs)

    def __repr__(self):
        axes = ["lon", "lat"] + [_.name for _ in self.axes]
        lon = self.center_skydir.data.lon.deg
        lat = self.center_skydir.data.lat.deg

        return (
            f"{self.__class__.__name__}\n\n"
            f"\taxes       : {axes}\n"
            f"\tshape      : {self.data_shape[::-1]}\n"
            f"\tndim       : {self.ndim}\n"
            f"\tframe      : {self.center_skydir.frame.name}\n"
            f"\tcenter     : {lon:.1f} deg, {lat:.1f} deg\n"
        )

    def __eq__(self, other):
        # check overall shape and axes compatibility
        if self.data_shape != other.data_shape:
            return False

        for axis, otheraxis in zip(self.axes, other.axes):
            if axis != otheraxis:
                return False

        # TODO: compare regions
        return True