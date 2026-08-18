import numpy as np

from .models import TableModel


def test_uint16_table_roundtrip(tmp_path):
    """Unsigned 16-bit table columns survive save without mutating the model."""
    path = tmp_path / "uint16_table.fits"
    n_rows = 10
    expected = np.arange(n_rows, dtype=np.uint16)

    with TableModel((n_rows,)) as model:
        model.table["uint16_column"] = expected
        model.save(path)
        np.testing.assert_array_equal(model.table["uint16_column"], expected)

    with TableModel(path) as model:
        np.testing.assert_array_equal(model.table["uint16_column"], expected)
