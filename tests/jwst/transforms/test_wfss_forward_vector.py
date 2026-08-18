import numpy as np
import pytest
from astropy.modeling.models import Polynomial1D, Polynomial2D
from numpy.testing import assert_allclose

from stdatamodels.jwst.transforms import models


def _wfss_model(direction):
    zero = Polynomial2D(0, c0_0=0.0)
    spatial_slope = Polynomial2D(1, c0_0=10.0, c1_0=0.01, c0_1=0.02)
    trace = [zero, spatial_slope]

    kwargs = {
        "orders": np.array([1]),
        "lmodels": [Polynomial1D(1, c0=1.0, c1=2.0)],
        "xmodels": [trace],
        "ymodels": [trace],
        "theta": 0.0,
        "sampling": 20,
    }
    if direction == "row":
        return models.NIRISSForwardRowGrismDispersion(**kwargs)
    return models.NIRISSForwardColumnGrismDispersion(**kwargs)


@pytest.mark.parametrize("direction", ["row", "column"])
def test_wfss_forward_supports_multiple_direct_image_pixels(direction):
    model = _wfss_model(direction)

    x0 = np.array([100.0, 200.0])
    y0 = np.array([50.0, 80.0])
    t = np.array([0.25, 0.75])
    order = np.array([1, 1])
    trace_scale = 10.0 + 0.01 * x0 + 0.02 * y0

    if direction == "row":
        x = x0 + trace_scale * t
        y = y0
    else:
        x = x0
        y = y0 + trace_scale * t

    direct_x, direct_y, wavelength, output_order = model(x, y, x0, y0, order)

    assert_allclose(direct_x, x0)
    assert_allclose(direct_y, y0)
    assert_allclose(wavelength, 1.0 + 2.0 * t, rtol=1e-7, atol=1e-7)
    assert_allclose(output_order, order)


def test_wfss_forward_broadcasts_single_direct_image_pixel():
    model = _wfss_model("row")

    x0 = 100.0
    y0 = 50.0
    t = np.array([0.25, 0.75])
    order = np.array([1, 1])
    trace_scale = 10.0 + 0.01 * x0 + 0.02 * y0
    x = x0 + trace_scale * t
    y = np.full_like(x, y0)

    direct_x, direct_y, wavelength, output_order = model.evaluate(x, y, x0, y0, order)

    assert_allclose(direct_x, np.full_like(x, x0))
    assert_allclose(direct_y, np.full_like(y, y0))
    assert_allclose(wavelength, 1.0 + 2.0 * t, rtol=1e-7, atol=1e-7)
    assert_allclose(output_order, order)
