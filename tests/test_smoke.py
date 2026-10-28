from golden_fighter import config


def test_config_paths_are_inside_repo():
    for path in (config.RAW_DATA_PATH, config.ANALYST_DIR, config.SCIENTIST_DIR):
        assert config.ROOT in path.parents


def test_seed_is_fixed():
    assert config.SEED == 42
