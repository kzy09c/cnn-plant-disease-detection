"""Execute the explanatory notebook with this environment; fresh training stays disabled."""
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
path = ROOT / "Tomato_CNN.ipynb"
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(notebook, km=manager, timeout=90, resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.write(notebook, path)
print("Executed all notebook cells successfully; saved real outputs and figures")
