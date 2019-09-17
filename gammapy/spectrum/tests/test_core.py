# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.units import Quantity
from gammapy.irf import EffectiveAreaTable, EnergyDispersion
from gammapy.modeling.models import PowerLaw, PowerLaw2, TableModel
from gammapy.spectrum import CountsSpectrum, SpectrumEvaluator
from gammapy.utils.energy import energy_logspace
from gammapy.utils.testing import (
    assert_quantity_allclose,
    mpl_plot_check,
    requires_dependency,
)


class TestCountsSpectrum:
    def setup(self):
        self.counts = [0, 0, 2, 5, 17, 3]
        self.bins = energy_logspace(1, 10, 7, "TeV")
        self.spec = CountsSpectrum(
            data=self.counts, energy_lo=self.bins[:-1], energy_hi=self.bins[1:]
        )

    def test_wrong_init(self):
        bins = energy_logspace(1, 10, 8, "TeV")
        with pytest.raises(ValueError):
            CountsSpectrum(data=self.counts, energy_lo=bins[:-1], energy_hi=bins[1:])

    @requires_dependency("matplotlib")
    def test_plot(self):
        with mpl_plot_check():
            self.spec.plot(show_energy=1 * u.TeV)

        with mpl_plot_check():
            self.spec.plot_hist()

        with mpl_plot_check():
            self.spec.peek()

    def test_io(self, tmpdir):
        filename = tmpdir / "test.fits"
        self.spec.write(filename)
        spec2 = CountsSpectrum.read(filename)
        assert_quantity_allclose(spec2.energy.edges, self.bins)

    def test_downsample(self):
        rebinned_spec = self.spec.downsample(2)
        assert rebinned_spec.energy.nbin == self.spec.energy.nbin / 2
        assert rebinned_spec.data.shape[0] == self.spec.data.shape[0] / 2
        assert rebinned_spec.total_counts == self.spec.total_counts

        idx = rebinned_spec.energy.coord_to_idx([2, 3, 5] * u.TeV)
        actual = rebinned_spec.data[idx]
        desired = [0, 7, 20]
        assert (actual == desired).all()


def get_test_cases():
    e_true = Quantity(np.logspace(-1, 2, 120), "TeV")
    e_reco = Quantity(np.logspace(-1, 2, 100), "TeV")

    aeff = EffectiveAreaTable.from_parametrization(e_true)

    exposure = CountsSpectrum(
        data=(aeff.data.data * 10 * u.h).to_value("cm2 s"),
        energy_lo=aeff.energy.edges[:-1],
        energy_hi=aeff.energy.edges[1:],
        unit="cm2 s"
    )

    exposure_2 = CountsSpectrum(
        data=(aeff.data.data * 30 * u.h).to_value("cm2 s"),
        energy_lo=aeff.energy.edges[:-1],
        energy_hi=aeff.energy.edges[1:],
        unit="cm2 s"
    )

    return [
        dict(model=PowerLaw(amplitude="1e2 TeV-1"), e_true=e_true, npred=999),
        dict(
            model=PowerLaw2(amplitude="1", emin="0.1 TeV", emax="100 TeV"),
            e_true=e_true,
            npred=1,
        ),
        dict(
            model=PowerLaw(amplitude="1e-11 TeV-1 cm-2 s-1"),
            exposure=exposure,
            npred=1448.05960,
        ),
        dict(
            model=PowerLaw(reference="1 GeV", amplitude="1e-11 GeV-1 cm-2 s-1"),
            exposure=exposure_2,
            npred=4.34417881,
        ),
        dict(
            model=PowerLaw(amplitude="1e-11 TeV-1 cm-2 s-1"),
            exposure=exposure,
            edisp=EnergyDispersion.from_gauss(
                e_reco=e_reco, e_true=e_true, bias=0, sigma=0.2
            ),
            npred=1437.450076,
        ),
        dict(
            model=TableModel(
                energy=[0.1, 0.2, 0.3, 0.4] * u.TeV,
                values=[4.0, 3.0, 1.0, 0.1] * u.Unit("TeV-1"),
            ),
            e_true=[0.1, 0.2, 0.3, 0.4] * u.TeV,
            npred=0.554513062,
        ),
    ]


@pytest.mark.parametrize("case", get_test_cases())
def test_counts_predictor(case):
    opts = case.copy()
    del opts["npred"]
    predictor = SpectrumEvaluator(**opts)
    actual = predictor.compute_npred().total_counts
    assert_allclose(actual, case["npred"])
