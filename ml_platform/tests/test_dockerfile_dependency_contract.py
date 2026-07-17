from pathlib import Path


def test_backend_dockerfile_uses_requirements_as_single_dependency_source():
    dockerfile = (
        Path(__file__).resolve().parents[2] / "docker" / "Dockerfile.backend"
    ).read_text()

    assert "pip install" in dockerfile
    assert "-r requirements.txt" in dockerfile
    assert "|| pip install" not in dockerfile


def test_backend_dockerfile_uses_reachable_python_mirror():
    dockerfile = (
        Path(__file__).resolve().parents[2] / "docker" / "Dockerfile.backend"
    ).read_text()

    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in dockerfile
