# Bedmock setup script.
#
# Usage for an existing app, from the folder with your code:
#   cd /path/to/app
#   source /path/to/bedmock/setup.sh
#
# This creates/activates /path/to/app/.venv and installs this Bedmock checkout
# into that venv.
#
# Usage for developing Bedmock itself, from the Bedmock checkout:
#   source setup.sh
#
# Optional overrides:
#   BEDMOCK_APP_DIR=/path/to/app          # venv and app requirements live here
#   BEDMOCK_VENV_DIR=/path/to/.venv       # custom venv location
#   BEDMOCK_SETUP_MODE=runtime|dev        # default: runtime for apps, dev for Bedmock
#   BEDMOCK_RUN_CHECKS=quick|full|false   # default: quick
#
# This script is intentionally sourced, not executed, so the virtualenv stays
# active in the caller's shell after setup completes.

(return 0 2>/dev/null) || {
  echo "ERROR: source this script instead of executing it."
  echo "Run: source ${BASH_SOURCE[0]:-$0}"
  exit 1
}

_BEDMOCK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CALLER_DIR="$(pwd)"
_APP_DIR_RAW="${BEDMOCK_APP_DIR:-}"
if [ -n "${_APP_DIR_RAW}" ]; then
  _APP_DIR="$(cd "${_APP_DIR_RAW}" 2>/dev/null && pwd)"
else
  _APP_DIR="${_CALLER_DIR}"
fi
if [ -z "${_APP_DIR}" ]; then
  echo "ERROR: app directory does not exist: ${_APP_DIR_RAW}"
  return 1
fi

_VENV_DIR="${BEDMOCK_VENV_DIR:-${_APP_DIR}/.venv}"
_SETUP_MODE_RAW="${BEDMOCK_SETUP_MODE:-}"
if [ -n "${_SETUP_MODE_RAW}" ]; then
  _MODE="${_SETUP_MODE_RAW}"
elif [ "${_APP_DIR}" = "${_BEDMOCK_ROOT}" ]; then
  _MODE="dev"
else
  _MODE="runtime"
fi
_RUN_CHECKS="${BEDMOCK_RUN_CHECKS:-quick}"
_FORCE_INSTALL="${BEDMOCK_FORCE_INSTALL:-false}"
_REQ_DIR_RAW="${BEDMOCK_REQUIREMENTS_DIR:-}"

case "$(uname -s 2>/dev/null)" in
  Linux*)               _OS="linux" ;;
  Darwin*)              _OS="macos" ;;
  MINGW*|MSYS*|CYGWIN*) _OS="windows" ;;
  *)                    _OS="unknown" ;;
esac

echo "-> App root:     ${_APP_DIR}"
echo "-> Bedmock root: ${_BEDMOCK_ROOT}"
echo "-> Venv:         ${_VENV_DIR}"
echo "-> Mode:         ${_MODE}"
[ "${_OS}" = "windows" ] && echo "-> Shell:        Git Bash on Windows"

_PYTHON=""
for _cand in python3 python py; do
  if command -v "${_cand}" >/dev/null 2>&1; then
    _PYTHON="${_cand}"
    break
  fi
done

if [ -z "${_PYTHON}" ]; then
  echo "ERROR: no Python interpreter found on PATH."
  echo "Install Python 3.10+ and re-source this script."
  return 1
fi

if [ ! -d "${_VENV_DIR}" ]; then
  echo "-> Creating virtualenv..."
  if ! "${_PYTHON}" -m venv "${_VENV_DIR}" 2>/tmp/bedmock-venv.err; then
    echo "ERROR: failed to create virtualenv:"
    sed 's/^/    /' /tmp/bedmock-venv.err
    echo "On Linux/WSL, install python3-venv and re-source this script."
    return 1
  fi
fi

if [ -f "${_VENV_DIR}/bin/activate" ]; then
  _ACTIVATE="${_VENV_DIR}/bin/activate"
elif [ -f "${_VENV_DIR}/Scripts/activate" ]; then
  _ACTIVATE="${_VENV_DIR}/Scripts/activate"
else
  echo "ERROR: could not find virtualenv activate script."
  echo "Remove ${_VENV_DIR} and re-source setup.sh."
  return 1
fi

# shellcheck disable=SC1090
source "${_ACTIVATE}"
echo "OK: virtualenv active: ${VIRTUAL_ENV}"

_NEEDS_INSTALL="true"
if [ "${_FORCE_INSTALL}" != "true" ]; then
  BEDMOCK_SETUP_ROOT="${_BEDMOCK_ROOT}" python -m pip show bedmock >/dev/null 2>&1 && \
  BEDMOCK_SETUP_ROOT="${_BEDMOCK_ROOT}" python - <<'PYCHECK'
import importlib.util
import os
from pathlib import Path

root = Path(os.environ["BEDMOCK_SETUP_ROOT"]).resolve()
spec = importlib.util.find_spec("bedmock")
if spec is None or spec.origin is None:
    raise SystemExit(1)
origin = Path(spec.origin).resolve()
if not origin.is_relative_to(root):
    raise SystemExit(1)
PYCHECK
  if [ $? -eq 0 ]; then
    _NEEDS_INSTALL="false"
  fi
fi

if [ "${_NEEDS_INSTALL}" = "true" ]; then
  echo "-> Installing Bedmock (${_MODE} mode)..."
  cd "${_BEDMOCK_ROOT}" || return 1
  if [ "${_MODE}" = "runtime" ]; then
    if ! python -m pip install --disable-pip-version-check -e .; then
      echo "-> Retrying editable install with --no-build-isolation..."
      if python -m pip install --disable-pip-version-check --no-build-isolation -e .; then
        _INSTALL_OK="true"
      else
        _INSTALL_OK="false"
      fi
    else
      _INSTALL_OK="true"
    fi
    if [ "${_INSTALL_OK}" != "true" ]; then
      echo "ERROR: pip install failed."
      echo "Common causes: no internet, proxy missing, or package index unavailable."
      echo "For offline setup, preinstall build tooling in the venv: setuptools>=77 and wheel."
      echo "If you are behind a proxy, set HTTPS_PROXY and re-source setup.sh."
      return 1
    fi
  elif [ "${_MODE}" = "dev" ]; then
    if ! python -m pip install --disable-pip-version-check -e ".[dev]"; then
      echo "-> Retrying editable install with --no-build-isolation..."
      if python -m pip install --disable-pip-version-check --no-build-isolation -e ".[dev]"; then
        _INSTALL_OK="true"
      else
        _INSTALL_OK="false"
      fi
    else
      _INSTALL_OK="true"
    fi
    if [ "${_INSTALL_OK}" != "true" ]; then
      echo "ERROR: pip install failed."
      echo "Common causes: no internet, proxy missing, or package index unavailable."
      echo "For offline setup, preinstall build tooling in the venv: setuptools>=77 and wheel."
      echo "If you are behind a proxy, set HTTPS_PROXY and re-source setup.sh."
      return 1
    fi
  else
    echo "ERROR: BEDMOCK_SETUP_MODE must be runtime or dev."
    return 1
  fi
else
  echo "OK: Bedmock is already installed from this checkout."
fi

if [ -n "${_REQ_DIR_RAW}" ]; then
  _REQ_DIR="$(cd "${_REQ_DIR_RAW}" 2>/dev/null && pwd)"
  if [ -z "${_REQ_DIR}" ]; then
    echo "ERROR: requirements directory does not exist: ${_REQ_DIR_RAW}"
    return 1
  fi
  if [ -f "${_REQ_DIR}/requirements.txt" ]; then
    echo "-> Installing requirements from ${_REQ_DIR}/requirements.txt..."
    if ! python -m pip install --disable-pip-version-check -r "${_REQ_DIR}/requirements.txt"; then
      echo "ERROR: requirements install failed."
      return 1
    fi
  else
    echo "-> No requirements.txt found in ${_REQ_DIR}; skipping app deps."
  fi
elif [ -f "${_APP_DIR}/requirements.txt" ] && [ "${_APP_DIR}" != "${_BEDMOCK_ROOT}" ]; then
  echo "-> Installing app requirements from ${_APP_DIR}/requirements.txt..."
  if ! python -m pip install --disable-pip-version-check -r "${_APP_DIR}/requirements.txt"; then
    echo "ERROR: app requirements install failed."
    return 1
  fi
fi

cd "${_APP_DIR}" || return 1

echo "-> Running import and CLI smoke checks..."
python - <<'PYCHECK'
import bedmock as boto3

client = boto3.client("bedrock-runtime")
assert client.meta.service_model.service_name == "bedrock-runtime"
print("OK: import bedmock as boto3")
print("OK: boto3.client('bedrock-runtime')")
PYCHECK
if [ $? -ne 0 ]; then
  echo "ERROR: Python import smoke check failed."
  return 1
fi

if [ -f ".env" ]; then
  _DOCTOR_CMD=(bedmock --env-file .env doctor --model-id "${BEDROCK_MODEL_ID:-}")
else
  _DOCTOR_CMD=(bedmock doctor --model-id "${BEDROCK_MODEL_ID:-us.amazon.nova-2-lite-v1:0}")
fi
if ! "${_DOCTOR_CMD[@]}"; then
  echo "ERROR: bedmock doctor failed to start."
  return 1
fi

if [ "${_RUN_CHECKS}" = "quick" ] && [ "${_APP_DIR}" = "${_BEDMOCK_ROOT}" ] && [ "${_MODE}" = "dev" ]; then
  echo "-> Running quick offline verification..."
  cd "${_BEDMOCK_ROOT}" || return 1
  if ! python -m pytest -q; then return 1; fi
  cd "${_APP_DIR}" || return 1
elif [ "${_RUN_CHECKS}" = "full" ] && [ "${_APP_DIR}" = "${_BEDMOCK_ROOT}" ] && [ "${_MODE}" = "dev" ]; then
  echo "-> Running full offline verification gates..."
  cd "${_BEDMOCK_ROOT}" || return 1
  if ! ruff check .; then return 1; fi
  if ! ruff format --check .; then return 1; fi
  if ! mypy bedmock; then return 1; fi
  if ! python -m pytest -q; then return 1; fi
  _BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/bedmock-build.XXXXXX")"
  if ! python -m build --no-isolation --outdir "${_BUILD_DIR}"; then return 1; fi
  if ! twine check "${_BUILD_DIR}"/*; then return 1; fi
  cd "${_APP_DIR}" || return 1
elif [ "${_RUN_CHECKS}" = "quick" ] && [ "${_APP_DIR}" != "${_BEDMOCK_ROOT}" ]; then
  echo "-> App setup quick check complete. Library test suite is skipped outside Bedmock root."
elif [ "${_RUN_CHECKS}" = "full" ] && [ "${_APP_DIR}" != "${_BEDMOCK_ROOT}" ]; then
  echo "-> Full Bedmock gates are only run from ${_BEDMOCK_ROOT}."
  echo "   App import and CLI smoke checks passed."
elif [ "${_RUN_CHECKS}" = "false" ]; then
  echo "-> Skipping verification gates because BEDMOCK_RUN_CHECKS=${_RUN_CHECKS}."
elif [ "${_MODE}" = "runtime" ]; then
  echo "-> Runtime mode selected; skipping dev-only gates."
else
  echo "ERROR: BEDMOCK_RUN_CHECKS must be quick, full, or false."
  return 1
fi

echo ""
echo "OK: Bedmock setup complete."
echo "Next:"
echo "  bedmock doctor"
echo "  pytest            # from Bedmock root in dev mode"
