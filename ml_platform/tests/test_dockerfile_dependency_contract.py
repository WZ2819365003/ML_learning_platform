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


def test_production_torch_wheel_does_not_compete_as_an_extra_index():
    requirements = (
        Path(__file__).resolve().parents[1] / "requirements.txt"
    ).read_text()

    assert "--extra-index-url" not in requirements
    assert "torch @ https://mirrors.aliyun.com/pytorch-wheels/cpu/" in requirements
    assert "#sha256=" in requirements
