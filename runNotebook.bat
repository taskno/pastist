@echo off
cd /d "%~dp0"

python\python.exe -m nbclassic ^
 --NotebookApp.notebook_dir="%~dp0." ^
 --NotebookApp.default_url="/notebooks/notebooks/MainWorkflowNotebook.ipynb" ^
 --ServerApp.open_browser=True ^
 --KernelManager.working_dir="%~dp0."

pause