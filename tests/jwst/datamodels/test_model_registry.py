from stdatamodels.jwst import datamodels


def test_model_registry_does_not_depend_on_all_ordering():
    expected = {name for name in datamodels.__all__ if name.endswith("Model")}

    assert set(datamodels._defined_models) == expected
    assert "open" not in datamodels._defined_models
    assert "read_metadata" not in datamodels._defined_models
