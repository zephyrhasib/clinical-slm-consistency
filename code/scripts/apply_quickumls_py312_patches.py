"""
Apply the two Python 3.12 compatibility patches required by QuickUMLS 1.4.2
in venv_metrics/. Idempotent — safe to re-run.

Patches:
  1. leveldb stub  : create venv_metrics/lib/python3.12/site-packages/leveldb.py
                     (quickumls.toolbox unconditionally imports leveldb at module
                     load even when database_backend='unqlite' is used — stub
                     prevents ImportError and raises only if the LevelDB code
                     path is actually executed)
  2. simstring imp : replace `import imp` (removed in Python 3.12) with importlib
                     in quickumls_simstring/simstring.py swig_import_helper()

Run with venv_metrics ACTIVE:
    source venv_metrics/bin/activate
    python scripts/apply_quickumls_py312_patches.py
"""
import re
import sys
import sysconfig
from pathlib import Path

# ── locate the active venv's site-packages ────────────────────────────────────
SITE = Path(sysconfig.get_paths()["purelib"])
print(f"Target site-packages: {SITE}")
if "venv_metrics" not in str(SITE):
    print(f"ERROR: must run with venv_metrics activated. Active site-packages: {SITE}")
    sys.exit(1)


# ── Patch 1 — leveldb stub ────────────────────────────────────────────────────
LEVELDB_STUB = SITE / "leveldb.py"
LEVELDB_CODE = '''\
"""leveldb stub injected by apply_quickumls_py312_patches.py
QuickUMLS 1.4.2 unconditionally does `import leveldb` even when the unqlite
backend is selected. We provide a stub that satisfies the import but raises
clearly if LevelDB code is actually used.
"""

class LevelDB:
    """Stub — instantiating LevelDB will raise. Never called when database_backend='unqlite'."""
    def __init__(self, path, *args, **kwargs):
        raise RuntimeError(
            "LevelDB stub: install libleveldb-dev + real leveldb to use this backend, "
            "or pass --database-backend unqlite to quickumls.install."
        )
'''
if LEVELDB_STUB.exists() and LEVELDB_STUB.read_text() == LEVELDB_CODE:
    print(f"[1/2] leveldb stub already present at {LEVELDB_STUB}")
else:
    LEVELDB_STUB.write_text(LEVELDB_CODE)
    print(f"[1/2] leveldb stub written : {LEVELDB_STUB}")


# ── Patch 2 — simstring imp → importlib ───────────────────────────────────────
SIMSTRING_PY = SITE / "quickumls_simstring" / "simstring.py"
if not SIMSTRING_PY.exists():
    print(f"ERROR: {SIMSTRING_PY} not found")
    sys.exit(1)

src      = SIMSTRING_PY.read_text()
MARKER   = "# PATCHED_FOR_PY312"
if MARKER in src:
    print(f"[2/2] simstring already patched — skip ({SIMSTRING_PY})")
else:
    # The original block (~21 lines) uses Python 2/3 dual-path via
    # `if version_info >= (2,6,0)` and calls `imp.find_module` / `imp.load_module`.
    # Replace the entire `from sys import version_info` … `del version_info` block
    # with an unconditional importlib-based loader.
    replacement = f"""{MARKER}
# imp module removed in Python 3.12; use importlib to load the _simstring extension.
def _load_simstring_extension():
    import importlib.util, os
    pkg_dir = os.path.dirname(__file__)
    for fname in os.listdir(pkg_dir):
        if fname.startswith('_simstring') and (fname.endswith('.so') or fname.endswith('.pyd')):
            spec = importlib.util.spec_from_file_location('_simstring', os.path.join(pkg_dir, fname))
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    import _simstring
    return _simstring

_simstring = _load_simstring_extension()
del _load_simstring_extension
"""
    new_src, count = re.subn(
        r"from sys import version_info\n.*?\ndel version_info\n",
        replacement,
        src,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        print(f"ERROR: could not locate version_info block in {SIMSTRING_PY}")
        sys.exit(1)
    SIMSTRING_PY.write_text(new_src)
    print(f"[2/2] simstring patched : {SIMSTRING_PY}")


# ── Smoke-import quickumls (without DB) to confirm no ImportError ─────────────
print("\nVerifying patched imports ...")
import importlib
for mod in ("leveldb", "quickumls_simstring.simstring", "quickumls"):
    try:
        m = importlib.import_module(mod)
        print(f"  [OK]   import {mod}")
    except Exception as e:
        print(f"  [FAIL] import {mod}: {type(e).__name__}: {e}")
        sys.exit(1)
print("\nAll patches applied. QuickUMLS imports cleanly.")
