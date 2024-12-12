from pathlib import Path

from scripts.miade import MakeConfig, Location, Source, URL


def test_parse_training_config_complete():
    reference_cfg_string = """
    model:
      location:
        path: models/test_model_0123456789.zip
    vocab:
      location:
        path: /path/to/file
    cdb:
      data:
        url: https://github.com///
    meta-models:
      - problems:
          location:
            path: /path/to/file
      - meds:
          data:
            path: /path/to/file

    """
    print(reference_cfg_string)

    cfg = MakeConfig.from_yaml_string(reference_cfg_string)
    reference_cfg = MakeConfig(
        model=Source(location=Location(location=Path("models/test_model_0123456789.zip"))),
        vocab=Source(location=Location(location=Path("/path/to/file"))),
        cdb=Source(data=Location(location=URL(path="https://github.com///"))),
        meta_models={
            "problems": Source(location=Location(location=Path("/path/to/file"))),
            "meds": Source(data=Location(location=Path("/path/to/file"))),
        },
    )
    print(reference_cfg)
    print(cfg)
    assert str(cfg) == str(reference_cfg)


def test_parse_training_config_incomplete():
    reference_cfg_string = """
    model:
      location:
        path: models/test_model_0123456789.zip
    vocab:
      location:
        path: /path/to/file

    """
    print(reference_cfg_string)

    cfg = MakeConfig.from_yaml_string(reference_cfg_string)
    reference_cfg = MakeConfig(
        model=Source(location=Location(location=Path("models/test_model_0123456789.zip"))),
        vocab=Source(location=Location(location=Path("/path/to/file"))),
        cdb=None,
        meta_models={},
    )
    print(reference_cfg)
    print(cfg)
    assert str(cfg) == str(reference_cfg)
