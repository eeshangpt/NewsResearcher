from pathlib import Path

import yaml

TRUSTED_OUTLETS_PATH = Path(__file__).resolve().parent.parent / "data" / "trusted_outlets.yaml"


def test_trusted_outlets_yaml_loads_as_nonempty_domain_to_tier_dict():
    outlets = yaml.safe_load(TRUSTED_OUTLETS_PATH.read_text())

    assert isinstance(outlets, dict)
    assert len(outlets) > 0
    for domain, tier in outlets.items():
        assert isinstance(domain, str) and domain
        assert isinstance(tier, str) and tier


def test_trusted_outlets_yaml_contains_expected_wire_and_major_tiers():
    outlets = yaml.safe_load(TRUSTED_OUTLETS_PATH.read_text())

    assert outlets["reuters.com"] == "wire"
    assert outlets["apnews.com"] == "wire"
    assert outlets["bbc.com"] == "major"
    assert outlets["theguardian.com"] == "major"
    assert set(outlets.values()) == {"wire", "major"}
