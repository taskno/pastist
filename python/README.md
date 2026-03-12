
## Python installation & configuration
1- Download an embedded python package (python version =3.12): https://www.python.org \
2- Install required packages:
 - Run commands under local python directory!
 - If you use external python installation, install requirements on the external python!

### Edit Path
Edit python312._pth as follows:
 ```bash
python312.zip
.
import site
Lib\site-packages
 ```
### Packages
- Install pip
 ```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```
- Install setuptools and tkinter
```bash
python -m pip install --target . setuptools
python -m pip install --target . tkinter-embed
```
- Install other requirements
```bash
- python -m pip install --no-cache-dir -r ..\requirements.txt
```
