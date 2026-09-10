from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata


PROJECT_ROOT = Path(SPECPATH).parent

datas = [
    (str(PROJECT_ROOT / "campus_agent" / "data"), "campus_agent/data"),
    (str(PROJECT_ROOT / "campus_agent" / "static"), "campus_agent/static"),
]
binaries = []
hiddenimports = []

for package in (
    "langchain",
    "langchain_core",
    "langchain_deepseek",
    "langgraph",
    "langgraph_checkpoint",
    "sentence_transformers",
):
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hiddenimports
    except Exception:
        # Optional semantic dependencies may intentionally be absent in a hashing-only build.
        pass

for distribution in (
    "campus-agent",
    "langchain",
    "langchain-core",
    "langchain-deepseek",
    "langgraph",
    "langgraph-checkpoint-sqlite",
    "sentence-transformers",
):
    try:
        datas += copy_metadata(distribution, recursive=True)
    except Exception:
        pass

hiddenimports += collect_submodules("campus_agent")

a = Analysis(
    [str(PROJECT_ROOT / "campus_agent" / "desktop_worker.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="campus-agent-worker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # The Rust parent creates this process with CREATE_NO_WINDOW, while a console
    # bootloader is still required so redirected stdin/stdout remain usable as the
    # NDJSON transport.
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="campus-agent-worker",
)
