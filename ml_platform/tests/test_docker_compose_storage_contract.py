from pathlib import Path

import yaml


def test_backend_storage_is_persistent():
    compose = yaml.safe_load(
        (Path(__file__).parents[2] / "docker" / "docker-compose.yml").read_text()
    )
    assert "ml_backend_storage:/app/storage" in compose["services"]["backend"]["volumes"]
    assert compose["volumes"]["ml_backend_storage"]["name"] == "ml_platform_backend_storage"
