import json

from helpers import analyze

PYPROJECT = """
[project]
name = "svc"
requires-python = ">=3.12"
dependencies = ["fastapi", "uvicorn"]
[project.scripts]
svc = "svc.cli:app"
[dependency-groups]
dev = ["pytest", "ruff"]
"""


def test_uv_project_gets_uv_commands(tmp_path):
    _, _, _, stack, _ = analyze(
        tmp_path, {"pyproject.toml": PYPROJECT, "uv.lock": "", "tests/test_a.py": ""}
    )
    c = stack.commands
    assert stack.package_managers == ["uv"]
    assert c.install == ["uv sync"]
    assert c.test == ["uv run pytest"]
    assert c.lint == ["uv run ruff check ."]
    assert ("uv run svc", "entry point `svc.cli:app`") in c.run
    assert c.prerequisites[0] == "Python `>=3.12`"


def test_poetry_project_gets_poetry_commands(tmp_path):
    text = '[tool.poetry]\nname="p"\n[tool.poetry.dependencies]\npython="^3.11"\n[tool.poetry.group.dev.dependencies]\npytest="*"\n'
    _, _, _, stack, _ = analyze(tmp_path, {"pyproject.toml": text, "poetry.lock": ""})
    assert stack.package_managers == ["Poetry"]
    assert stack.commands.install == ["poetry install"]
    assert stack.commands.test == ["poetry run pytest"]


def test_plain_pyproject_falls_back_to_pip_editable_install(tmp_path):
    _, _, _, stack, _ = analyze(tmp_path, {"pyproject.toml": '[project]\nname = "p"\n'})
    assert stack.package_managers == ["pip"]
    assert stack.commands.install[-1] == "pip install -e ."


def test_requirements_only_project_installs_runtime_requirements_first(tmp_path):
    files = {"requirements.txt": "flask\n", "requirements-dev.txt": "pytest\n"}
    _, _, _, stack, _ = analyze(tmp_path, files)
    assert stack.commands.install[-2:] == [
        "pip install -r requirements.txt",
        "pip install -r requirements-dev.txt",
    ]


def test_requirements_file_is_ignored_when_pyproject_exists(tmp_path):
    files = {"pyproject.toml": '[project]\nname = "p"\n', "requirements.txt": "flask\n"}
    _, _, _, stack, _ = analyze(tmp_path, files)
    assert stack.ecosystems == ["python"]
    assert stack.commands.install[-1] == "pip install -e ."


def test_package_main_becomes_the_run_command_when_no_scripts(tmp_path):
    files = {
        "pyproject.toml": '[project]\nname = "p"\n',
        "uv.lock": "",
        "src/pkg/__main__.py": "",
    }
    _, _, _, stack, _ = analyze(tmp_path, files)
    assert ("uv run python -m pkg", "package entry point") in stack.commands.run


def test_fastapi_app_object_is_located_for_uvicorn(tmp_path):
    files = {
        "pyproject.toml": PYPROJECT,
        "uv.lock": "",
        "src/svc/web.py": "from fastapi import FastAPI\napp = FastAPI()\n",
    }
    _, _, _, stack, _ = analyze(tmp_path, files)
    assert (
        "uv run uvicorn svc.web:app --reload",
        "FastAPI development server",
    ) in stack.commands.run


def test_streamlit_script_is_located_by_import_not_just_filename(tmp_path):
    base = '[project]\nname="p"\ndependencies=["streamlit"]\n'
    files = {
        "pyproject.toml": base,
        "uv.lock": "",
        "src/p/app.py": "import streamlit as st\n",
        "other/app.py": "print(1)\n",
    }
    _, _, _, stack, _ = analyze(tmp_path, files)
    run = [command for command, _ in stack.commands.run]
    assert run == ["uv run streamlit run src/p/app.py"]


def _node(extra: dict, files: dict | None = None) -> dict:
    package = {
        "name": "web",
        "scripts": {
            "dev": "vite",
            "build": "vite build",
            "test": "vitest",
            "lint": "eslint .",
        },
        **extra,
    }
    return {"package.json": json.dumps(package), **(files or {})}


def test_node_manager_is_detected_from_lockfile(tmp_path):
    for lockfile, manager in [
        ("pnpm-lock.yaml", "pnpm"),
        ("yarn.lock", "yarn"),
        ("package-lock.json", "npm"),
    ]:
        root = tmp_path / manager
        root.mkdir()
        _, _, _, stack, _ = analyze(root, _node({}, {lockfile: ""}))
        assert stack.package_managers == [manager]
        assert stack.commands.install == [f"{manager} install"]


def test_declared_package_manager_beats_lockfile(tmp_path):
    _, _, _, stack, _ = analyze(
        tmp_path, _node({"packageManager": "pnpm@9"}, {"package-lock.json": ""})
    )
    assert stack.package_managers == ["pnpm"]


def test_node_scripts_use_the_right_invocation_form(tmp_path):
    _, _, _, stack, _ = analyze(tmp_path, _node({}, {"package-lock.json": ""}))
    c = stack.commands
    assert c.run == [("npm run dev", "`vite`")]
    assert c.build == ["npm run build"]
    assert c.test == ["npm test"]
    assert c.lint == ["npm run lint"]


def test_rust_and_go_commands(tmp_path):
    rust = tmp_path / "rust"
    _, _, _, stack, _ = analyze(
        rust, {"Cargo.toml": '[package]\nname="c"\n', "src/main.rs": "fn main(){}"}
    )
    assert stack.commands.run == [("cargo run", "default binary")]
    assert stack.commands.test == ["cargo test"]

    go = tmp_path / "go"
    files = {
        "go.mod": "module x/y\n\ngo 1.22\n",
        "cmd/api/main.go": "package main",
        "cmd/worker/main.go": "package main",
    }
    _, _, _, stack, _ = analyze(go, files)
    assert [c for c, _ in stack.commands.run] == [
        "go run ./cmd/api",
        "go run ./cmd/worker",
    ]
    assert stack.commands.prerequisites == ["[Go](https://go.dev/dl/) 1.22"]


def test_technologies_are_grouped_by_category_and_deduplicated(tmp_path):
    files = {
        "package.json": json.dumps(
            {
                "dependencies": {"react": "1", "next": "1"},
                "devDependencies": {"vitest": "1", "jest": "1"},
            }
        ),
        "api/pyproject.toml": '[project]\nname="api"\ndependencies=["fastapi", "pydantic"]\n',
    }
    _, _, _, stack, _ = analyze(tmp_path, files)
    assert stack.technologies == {
        "Web framework": ["FastAPI", "Next.js"],
        "UI": ["React"],
        "Data": ["Pydantic"],
        "Testing": ["Jest", "Vitest"],
    }


def test_go_module_prefix_matches_versioned_paths(tmp_path):
    _, _, _, stack, _ = analyze(
        tmp_path,
        {
            "go.mod": "module m\n\ngo 1.22\n\nrequire github.com/gin-gonic/gin/v2 v2.0.0\n"
        },
    )
    assert stack.technologies == {"Web framework": ["Gin"]}


def test_empty_repo_yields_no_stack(tmp_path):
    _, _, _, stack, _ = analyze(tmp_path, {"notes.txt": "hi"})
    assert (
        stack.ecosystems == []
        and stack.technologies == {}
        and stack.commands.install == []
    )
