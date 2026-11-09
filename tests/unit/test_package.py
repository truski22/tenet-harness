import importlib
import pkgutil

import tenet_harness


def test_every_module_imports():
    names = [m.name for m in pkgutil.walk_packages(tenet_harness.__path__, "tenet_harness.")]
    assert len(names) > 15
    for name in names:
        importlib.import_module(name)


def test_public_api_is_complete():
    for name in tenet_harness.__all__:
        assert hasattr(tenet_harness, name), name
