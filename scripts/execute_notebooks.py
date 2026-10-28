#!/usr/bin/env python3
"""Execute and save all tutorial notebooks with real Jupyter kernel outputs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
# Keep generated kernel/cache files out of user configuration directories.
CACHE=Path(tempfile.gettempdir()) / "cell_motility_notebooks"
for key,part in (("JUPYTER_RUNTIME_DIR","runtime"),("IPYTHONDIR","ipython"),
                 ("MPLCONFIGDIR","matplotlib"),("XDG_CACHE_HOME","cache")):
    directory=CACHE / part;directory.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault(key,str(directory))

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager


class LocalKernelManager(AsyncKernelManager):
    """Use local Unix sockets on POSIX; no TCP listener is needed."""
    def __init__(self, **kwargs):
        if os.name == "posix":
            kwargs.setdefault("transport", "ipc")
            # macOS limits Unix-socket paths to roughly 104 bytes; TMPDIR can
            # already be long, so keep only the IPC endpoint under /tmp.
            kwargs.setdefault("ip", "/tmp/cm-k-"+uuid.uuid4().hex[:16])
        super().__init__(**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern",default="*.ipynb")
    args=parser.parse_args();manifest=[]
    for path in sorted((ROOT / "notebooks").glob(args.pattern)):
        print(f"Executing {path.name}",flush=True)
        notebook=nbformat.read(path,as_version=4)
        client=NotebookClient(notebook,timeout=180,kernel_name="python3",allow_errors=False,
                              kernel_manager_class=LocalKernelManager,
                              resources={"metadata":{"path":str(ROOT)}},record_timing=True)
        start=time.perf_counter();client.execute();elapsed=time.perf_counter()-start
        nbformat.write(notebook,path)
        manifest.append({"notebook":path.name,"seconds":elapsed,
            "code_cells":sum(c.cell_type=="code" for c in notebook.cells),
            "executed_code_cells":sum(c.cell_type=="code" and c.execution_count is not None for c in notebook.cells),
            "errors":sum(o.output_type=="error" for c in notebook.cells if c.cell_type=="code" for o in c.outputs)})
        print(f"Completed in {elapsed:.1f}s",flush=True)
    out=ROOT / "results" / "synthetic_v1";out.mkdir(parents=True,exist_ok=True)
    (out / "notebook_execution.json").write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
