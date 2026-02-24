# aurora

conda create --name aurora python==3.11
conda activate aurora
pip install numpy pandas torch microsoft-aurora


## Notes
- SLP should be in Pascals not hectopascals! Do the conversion inside run-aurora.py and convert back before writing it to disk.
- Don't manually normalize your data. The aurora library has a normalization function (and de-normalization) which are called inside run-aurora.py.
