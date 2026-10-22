"""Execute the explanatory notebook with this environment; fresh training stays disabled."""
import argparse
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("notebook", nargs="?", default="Tomato_CNN.ipynb",
                    help="Notebook filename: Tomato_CNN.ipynb, Bangladesh_CNN.ipynb or PlantVillage_Bangladesh_Finetuning.ipynb")
args = parser.parse_args()
path = ROOT / args.notebook
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(notebook, km=manager, timeout=90, resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.write(notebook, path)
print("Executed all notebook cells successfully; saved real outputs and figures")
