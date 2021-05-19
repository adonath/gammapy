# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
import astropy.units as u
from gammapy.data import GTI
from gammapy.maps import MapAxis, WcsNDMap, WcsGeom
from gammapy.modeling.models import SkyModel, PowerLawSpectralModel, PointSpatialModel, LogParabolaSpectralModel
from gammapy.estimators import FluxMaps
from gammapy.utils.testing import mpl_plot_check, requires_dependency


@pytest.fixture(scope="session")
def logpar_reference_model():
    logpar = LogParabolaSpectralModel(amplitude="2e-12 cm-2s-1TeV-1", alpha=1.5, beta=0.5)
    return SkyModel(spatial_model=PointSpatialModel(), spectral_model=logpar)


@pytest.fixture(scope="session")
def wcs_flux_map():
    energy_axis = MapAxis.from_energy_bounds(0.1, 10, 2, unit='TeV')
    spectral_model = PowerLawSpectralModel(index=2)

    geom = WcsGeom.create(npix=10, frame='galactic', axes=[energy_axis])
    map_dict = spectral_model.to_reference_flux_maps(geom=geom)

    map_dict["norm"] = WcsNDMap.from_geom(geom=geom)
    map_dict["norm"].data += 1.0

    map_dict["norm_err"] = WcsNDMap.from_geom(geom=geom)
    map_dict["norm_err"].data += 0.1

    map_dict["norm_errp"] = WcsNDMap.from_geom(geom=geom)
    map_dict["norm_errp"].data += 0.2

    map_dict["norm_errn"] = WcsNDMap.from_geom(geom=geom)
    map_dict["norm_errn"].data += 0.2

    map_dict["norm_ul"] = WcsNDMap.from_geom(geom=geom)
    map_dict["norm_ul"].data += 2.0

    # Add another map
    map_dict["sqrt_ts"] = WcsNDMap.from_geom(geom=geom)
    map_dict["sqrt_ts"].data += 1.0

    # Add another map
    map_dict["ts"] = WcsNDMap.from_geom(geom=geom)
    map_dict["ts"].data[1] += 3.0

    return map_dict


@pytest.fixture(scope="session")
def partial_wcs_flux_map():
    energy_axis = MapAxis.from_energy_bounds(0.1, 10, 2, unit='TeV')

    spectral_model = PowerLawSpectralModel(index=2)

    geom = WcsGeom.create(npix=10, frame='galactic', axes=[energy_axis])
    map_dict = spectral_model.to_reference_flux_maps(geom=geom)

    map_dict["norm"]= WcsNDMap.from_geom(geom=geom)
    map_dict["norm"].data += 1.0

    map_dict["norm_err"] = WcsNDMap.from_geom(geom=geom)
    map_dict["norm_err"].data += 0.1

    # Add another map
    map_dict["sqrt_ts"] = WcsNDMap.from_geom(geom=geom)
    map_dict["sqrt_ts"].data += 1.0

    return map_dict


def test_flux_map_properties(wcs_flux_map):
    fluxmap = FluxMaps.from_dict(maps=wcs_flux_map)

    assert_allclose(fluxmap.dnde.data[:, 0, 0], [1e-11, 1e-13])
    assert_allclose(fluxmap.dnde_err.data[:, 0, 0], [1e-12, 1e-14])
    assert_allclose(fluxmap.dnde_err.data[:, 0, 0], [1e-12, 1e-14])
    assert_allclose(fluxmap.dnde_errn.data[:, 0, 0], [2e-12, 2e-14])
    assert_allclose(fluxmap.dnde_errp.data[:, 0, 0], [2e-12, 2e-14])
    assert_allclose(fluxmap.dnde_ul.data[:, 0, 0], [2e-11, 2e-13])

    assert_allclose(fluxmap.flux.data[:, 0, 0], [9e-12, 9e-13])
    assert_allclose(fluxmap.flux_err.data[:, 0, 0], [9e-13, 9e-14])
    assert_allclose(fluxmap.flux_errn.data[:, 0, 0], [18e-13, 18e-14])
    assert_allclose(fluxmap.flux_errp.data[:, 0, 0], [18e-13, 18e-14])
    assert_allclose(fluxmap.flux_ul.data[:, 0, 0], [18e-12, 18e-13])

    assert_allclose(fluxmap.eflux.data[:, 0, 0], [2.302585e-12, 2.302585e-12])
    assert_allclose(fluxmap.eflux_err.data[:, 0, 0], [2.302585e-13, 2.302585e-13])
    assert_allclose(fluxmap.eflux_errp.data[:, 0, 0], [4.60517e-13, 4.60517e-13])
    assert_allclose(fluxmap.eflux_errn.data[:, 0, 0], [4.60517e-13, 4.60517e-13])
    assert_allclose(fluxmap.eflux_ul.data[:, 0, 0], [4.60517e-12, 4.60517e-12])

    assert_allclose(fluxmap.e2dnde.data[:, 0, 0], [1e-12, 1e-12])
    assert_allclose(fluxmap.e2dnde_err.data[:, 0, 0], [1e-13, 1e-13])
    assert_allclose(fluxmap.e2dnde_errn.data[:, 0, 0], [2e-13, 2e-13])
    assert_allclose(fluxmap.e2dnde_errp.data[:, 0, 0], [2e-13, 2e-13])
    assert_allclose(fluxmap.e2dnde_ul.data[:, 0, 0], [2e-12, 2e-12])

    assert_allclose(fluxmap.sqrt_ts.data, 1)
    assert_allclose(fluxmap.ts.data[:, 0, 0], [0, 3])


def test_flux_map_str(wcs_flux_map):
    fluxmap = FluxMaps(data=wcs_flux_map)

    fm_str = fluxmap.__str__()

    assert "WcsGeom" in fm_str
    assert "errn" in fm_str
    assert "sqrt_ts" in fm_str


@pytest.mark.parametrize("sed_type", ["likelihood", "dnde", "flux", "eflux", "e2dnde"])
def test_flux_map_read_write(tmp_path, wcs_flux_map, sed_type):
    fluxmap = FluxMaps.from_dict(maps=wcs_flux_map, sed_type="likelihood")

    fluxmap.write(tmp_path / "tmp.fits", sed_type=sed_type)
    new_fluxmap = FluxMaps.read(tmp_path / "tmp.fits")

    assert_allclose(new_fluxmap.norm.data[:, 0, 0], [1, 1])
    assert_allclose(new_fluxmap.norm_err.data[:, 0, 0], [0.1, 0.1])
    assert_allclose(new_fluxmap.norm_errn.data[:, 0, 0], [0.2, 0.2])
    assert_allclose(new_fluxmap.norm_ul.data[:, 0, 0], [2, 2])

    # check existence and content of additional map
    assert_allclose(new_fluxmap.data["sqrt_ts"].data, 1.0)


@pytest.mark.parametrize("sed_type", ["likelihood", "dnde", "flux", "eflux", "e2dnde"])
def test_partial_flux_map_read_write(tmp_path, partial_wcs_flux_map, sed_type):
    fluxmap = FluxMaps.from_dict(maps=partial_wcs_flux_map)

    fluxmap.write(tmp_path / "tmp.fits", sed_type=sed_type)
    new_fluxmap = FluxMaps.read(tmp_path / "tmp.fits")

    # check existence and content of additional map
    assert_allclose(new_fluxmap.data["sqrt_ts"].data, 1.0)

    # the TS map shouldn't exist
    with pytest.raises(KeyError):
        new_fluxmap.data["ts"]


def test_flux_map_read_write_gti(tmp_path, partial_wcs_flux_map):
    start = u.Quantity([1, 2], "min")
    stop = u.Quantity([1.5, 2.5], "min")
    gti = GTI.create(start, stop)

    fluxmap = FluxMaps.from_dict(maps=partial_wcs_flux_map, gti=gti)

    fluxmap.write(tmp_path / "tmp.fits", sed_type='dnde')
    new_fluxmap = FluxMaps.read(tmp_path / "tmp.fits")

    assert len(new_fluxmap.gti.table) == 2
    assert_allclose(gti.table["START"], start.to_value("s"))


@requires_dependency("matplotlib")
def test_get_flux_point(wcs_flux_map):
    fluxmap = FluxMaps.from_dict(maps=wcs_flux_map)

    coord = SkyCoord(0., 0., unit="deg", frame="galactic")
    fp = fluxmap.get_flux_points(coord)

    assert_allclose(fp.table["e_min"], [0.1, 1.0])
    assert_allclose(fp.table["norm"], [1, 1])
    assert_allclose(fp.table["norm_err"], [0.1, 0.1])
    assert_allclose(fp.table["norm_errn"], [0.2, 0.2])
    assert_allclose(fp.table["norm_errp"], [0.2, 0.2])
    assert_allclose(fp.table["norm_ul"], [2, 2])
    assert_allclose(fp.table["sqrt_ts"], [1, 1])
    assert_allclose(fp.table["ts"], [0, 3])

    assert_allclose(fp.dnde, [1e-11, 1e-13] * u.Unit("TeV-1 cm-2 s-1"))
    assert fp.dnde.unit == "cm-2s-1TeV-1"

    with mpl_plot_check():
        fp.plot()


def test_get_flux_point_missing_map(wcs_flux_map):
    other_data = wcs_flux_map.copy()
    other_data.pop("norm_errn")
    other_data.pop("norm_errp")
    fluxmap = FluxMaps.from_dict(maps=other_data)

    coord = SkyCoord(0., 0., unit="deg", frame="galactic")
    fp = fluxmap.get_flux_points(coord)

    assert_allclose(fp.table["e_min"], [0.1, 1.0])
    assert_allclose(fp.table["norm"], [1, 1])
    assert_allclose(fp.table["norm_err"], [0.1, 0.1])
    assert_allclose(fp.table["norm_ul"], [2, 2])
    assert "norm_errn" not in fp.table.columns


def test_flux_map_from_dict_inconsistent_units(wcs_flux_map):
    ref_map = FluxMaps.from_dict(maps=wcs_flux_map)
    map_dict = dict()
    map_dict["eflux"] = ref_map.eflux
    map_dict["eflux"].quantity = map_dict["eflux"].quantity.to("keV/m2/s")
    map_dict["eflux_err"] = ref_map.eflux_err
    map_dict["eflux_err"].quantity = map_dict["eflux_err"].quantity.to("keV/m2/s")

    flux_map = FluxMaps.from_dict(map_dict, sed_type="eflux")

    assert_allclose(flux_map.norm.data, 1)
    assert flux_map.norm.unit == ""
    assert_allclose(flux_map.norm_err.data, 0.1)
    assert flux_map.norm_err.unit == ""

