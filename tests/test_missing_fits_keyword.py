from astropy.io import fits

from stdatamodels.jwst import datamodels


def test_deleted_fits_keyword_clears_embedded_metadata(tmp_path):
    """A deleted FITS keyword must not be restored from stale embedded ASDF metadata."""
    source = tmp_path / "source.fits"
    modified = tmp_path / "modified.fits"

    with datamodels.ImageModel() as model:
        model.meta.instrument.name = "NIRCAM"
        model.save(source)

    with fits.open(source) as hdulist:
        del hdulist[0].header["INSTRUME"]
        hdulist.writeto(modified)

    with datamodels.open(modified) as model:
        assert model.meta.instrument.name is None
