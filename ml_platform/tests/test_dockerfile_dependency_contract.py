from pathlib import Path


def test_backend_dockerfile_uses_requirements_as_single_dependency_source():
    dockerfile = (
        Path(__file__).resolve().parents[2] / "docker" / "Dockerfile.backend"
    ).read_text()

    assert "pip install" in dockerfile
    assert "COPY ml_platform/requirements.txt /app/requirements.txt" in dockerfile
    assert "requirements.txt > /tmp/requirements-docker.txt" in dockerfile
    assert "-r /tmp/requirements-docker.txt" in dockerfile
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


def test_backend_build_parallelizes_and_verifies_the_torch_wheel():
    dockerfile = (
        Path(__file__).resolve().parents[2] / "docker" / "Dockerfile.backend"
    ).read_text()

    assert "aria2c" in dockerfile
    assert "--max-connection-per-server=16" in dockerfile
    assert "sha256sum -c -" in dockerfile
    assert "file:///tmp/torch.whl#sha256=" in dockerfile
