# Tests — Path handling strategy

This note explains how Python imports and file paths work across the project's assignment modules and their tests.

---

## The core problem

The project has two structural features that make path handling non-trivial:

1. **Notebook utils modules** (`w1/utils.py`, `w2/utils.py`, `w3/utils.py`) live in their own directories and need to import the shared ``setting.py`` from the project root.

2. **Pytest creates its own package namespace** — when it discovers tests in e.g. ``tests/w3/``, it treats that directory as the ``tests.w3`` package. If a root-level ``w3/`` package also exists, ``from w3.chroma_store import ChromaStore`` can fail because ``tests.w3`` shadows the real ``w3``.

---

## How paths are resolved

### Setting.py (the shared configuration)

`setting.py` anchors itself to the project root using its own location:

```python
PROJECT_ROOT = Path(__file__).resolve().parent
```

Relative paths from ``config.yaml`` are resolved against ``PROJECT_ROOT`` via the ``resolve_path()`` helper. The resulting config object holds absolute ``Path`` objects for data files and model directories.

### Assignment modules (w1/utils.py, w2/utils.py)

These modules are designed to be imported from Jupyter notebooks running in the project root directory. They use:

```python
PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from setting import config
```

This works when the notebook kernel's working directory is the project root, but it **modifies ``sys.path``** — a side effect that needs to be stubbed or isolated in tests.

### W3 (newer module)

The W3 adaptation avoids ``sys.path`` manipulation. Instead, ``tests/w3/test_chroma_store.py`` loads the production module via ``importlib``:

```python
import importlib.util
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "w3.chroma_store",
    _PROJECT_ROOT / "w3" / "chroma_store.py",
)
```

This is the **preferred pattern** — it avoids both ``sys.path`` pollution and pytest namespace conflicts.

### Tests (tests/w1_w2/)

The ``tests/w1_w2/support.py`` module contains an ``import_from_path()`` helper that uses the same ``importlib`` technique:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def import_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
```

Tests call it like:

```python
module = import_from_path("w1_behavior_utils", PROJECT_ROOT / "w1" / "utils.py")
```

---

## Summary of patterns

| Component | How it finds the project root | Import method | Notes |
|-----------|-------------------------------|---------------|-------|
| ``setting.py`` | ``Path(__file__).resolve().parent`` | Standard import | Always correct — module knows its own location |
| ``w1/utils.py``, ``w2/utils.py`` | ``Path.cwd().parent`` | ``sys.path`` insert + standard import | Depends on cwd; side-effect on ``sys.path`` |
| ``tests/w1_w2/`` | ``Path(__file__).resolve().parents[2]`` | ``importlib`` via ``import_from_path()`` | No namespace conflicts; no ``sys.path`` pollution |
| ``tests/w3/test_chroma_store.py`` | ``Path(__file__).resolve().parents[2]`` | ``importlib`` via ``spec_from_file_location`` | Preferred for future modules |
| Jupyter notebooks | ``Path.cwd().parent`` | ``sys.path`` insert + standard import | Implicit — assumes kernel cwd is project root |

---

## Guidelines for new code

1. **In test files**: use ``Path(__file__).resolve().parents[n]`` to reach the project root, then load production modules via ``importlib``. Never rely on ``sys.path`` manipulation or ``PYTHONPATH``.

2. **In production modules**: prefer absolute imports within the same package. For cross-package imports (e.g. importing ``setting.py`` from a ``w3/`` module), use ``importlib`` or ask the caller to inject the dependency.

3. **When you need ``sys.path``**: document why. The ``w1/`` and ``w2/`` utils modules have a legitimate reason (Jupyter compatibility), but newer modules should not follow that pattern.