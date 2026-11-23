# Installation Problems

Each entry is a **symptom**, its **cause**, and a **solution**.

## `ModuleNotFoundError: No module named 'menos'`

| | |
|---|---|
| **Symptom** | `import menos` fails |
| **Cause** | MENOS is not installed in the Python environment you are running, most often because the virtual environment is not the one in use |
| **Solution** | Run the interpreter from the environment you installed into (`.venv/bin/python`), or activate it. Check with `python -c "import sys; print(sys.executable)"` |

## `ImportError` mentioning `_core`

| | |
|---|---|
| **Symptom** | `import menos` fails with an `ImportError` about `menos._core` |
| **Cause** | The native extension did not build, or was built for a different Python version than the one importing it |
| **Solution** | Reinstall into the environment you import from (for example `uv venv .venv --python 3.14`). An extension built for one Python version cannot load in another. On Windows, `DLL load failed` means `libopenblas.dll` is missing next to `_core*.pyd` in the installed `menos` package; reinstall after running `scripts\fetch-openblas-windows.ps1` |

## `ModuleNotFoundError: No module named 'qiskit'`

| | |
|---|---|
| **Symptom** | `from menos.qiskit import MENOSBackend` fails |
| **Cause** | Qiskit is an optional extra |
| **Solution** | Install with the extra: `pip install "menos[qiskit]"` or `uv add "menos[qiskit]"` (from a source checkout: `uv pip install --python .venv/bin/python ".[qiskit]"`). The [native API](/build/native-api) works without it |

## The installer says the package requires a different Python

| | |
|---|---|
| **Symptom** | `pip` or `uv` refuses to install, naming a Python version requirement |
| **Cause** | `menos` requires Python `>=3.10`, and PyPI has wheels only for CPython 3.10–3.14 on Linux x86-64 and Windows x64 |
| **Solution** | Use Python 3.10–3.14 (`uv python install 3.14` installs one). No source distribution is published; on other platforms [install from source](/get-started/installation#install-from-source) |

## Qiskit version problems

| | |
|---|---|
| **Symptom** | `MENOSBackend` or `MENOSEstimator` raise errors from inside Qiskit, or behave unexpectedly |
| **Cause** | The `qiskit` extra pins **2.5.2**. A different installed version is untested |
| **Solution** | `uv pip install --python .venv/bin/python "qiskit==2.5.2"` |

## The build cannot download its dependencies

| | |
|---|---|
| **Symptom** | The build fails while configuring, with a network or `git` error |
| **Cause** | When building from source, CMake fetches GoogleTest and nlohmann/json from GitHub at configure time, even for a Python install. They are used only by the test suite, but the build needs them |
| **Solution** | Ensure `git` is installed and `github.com` is reachable (check proxy settings). Offline builds are not supported |

## A stale build after changing options

| | |
|---|---|
| **Symptom** | A rebuild ignores a new CMake option, for example turning CUDA on |
| **Cause** | The previous configuration is cached in the build directory (`build/py-…` for Python installs) |
| **Solution** | Delete that build directory and install again |

## Installation succeeded but examples fail

| | |
|---|---|
| **Symptom** | An example in the repository fails immediately |
| **Cause** | Usually the `qiskit` extra is missing, or you are running a different interpreter |
| **Solution** | Run [Verify Installation](/get-started/verify-installation) |

## Python exits with an access violation on Windows

| | |
|---|---|
| **Symptom** | With the 1.3.0 Windows wheel, a script that imports `qiskit_aer` before `menos` exits with status 3221225477 (0xC0000005) on a job with a time budget |
| **Cause** | `qiskit-aer` ships its own `libopenblas.dll`. Once it is loaded, Windows gives MENOS that library in place of its own, and the first job that reaches LAPACK crashes |
| **Solution** | Import `menos` before `qiskit_aer`, or use version 1.3.1 or later, whose wheels carry their libraries under hashed names |

## See also

[Build Errors](/troubleshooting/build-errors) · [Requirements](/get-started/requirements)
