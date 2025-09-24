# Xwakes

Python package for wakes and impedances handling.

## Installation

Under a conda environment with Python 3.8+ it is simply installed via PyPI by doing `pip install xwakes`

## IW2D coupling

This section describes how to couple Xwakes to IW2D using the executables obtained compiling the C++ code.
When the Python interface of IW2D will be completed this will not be needed anymore.

To begin with, some folders need to be created in the user's home directory.
This can be automatically done by running the following command after Xwakes is installed:
```
python -c 'import xwakes; xwakes.initialize_pywit_directory()'
```
The IW2D executable are produced by following the [IW2D readme]([https://gitlab.cern.ch/IRIS/IW2D/-/tree/master?ref_type=heads)](https://gitlab.cern.ch/IRIS/IW2D/).
After this procedure is completed the following executable files will be created in `/path/to/iw2d/IW2D/cpp/`:
   * `flatchamber.x`
   * `roundchamber.x`
   * `wake_flatchamber.x`
   * `wake_roundchamber.x`

These files have to be copied in the newly created folder with the command

```
cp /path/to/iw2d/IW2D/cpp/*.x ~/pywit/IW2D/bin
```
Now Xwakes can be used to launch IW2D calculations.
