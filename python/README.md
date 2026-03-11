## Package installation
1- Download an embedded python package (python version <=3.12): https://www.python.org \
2- Install required packages:
 - Run commands under local python directory!
 - If you use external python installation, install requirements on the external python!

### Commands
- curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
- python get-pip.py

Using requirements.txt file:\
- python -m pip install --no-cache-dir -r ..\requirements.txt

Explicitly install following packages: \
- python.exe -m pip install --target . setuptools
- python.exe -m pip install --target . tkinter-embed
