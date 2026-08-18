import numpy as np

from .models import TableModel


def test_uint16_table_roundtrip(tmp_path):
    """Unsigned 16-bit table columns survive save without mutating the model."""
    path = tmp_path / "uint16_table.fits"
    n_rows = 10
    expected = np.arange(n_rows, dtype=np.uint16)

    with TableModel((n_rows,)) as model:
        # Keep the unrelated legacy int8/logical column away from Astropy's
        # NULL-value path so this regression isolates unsigned integer I/O.
        model.table["int8_column"] = np.ones(n_rows, dtype=np.int8)
        model.table["uint16_column"] = expected
        model.save(path)
        np.testing.assert_array_equal(model.table["uint16_column"], expected)

    with TableModel(path) as model:
        np.testing.assert_array_equal(model.table["uint16_column"], expected)
