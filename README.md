# PyWIT
####Python Wake and Impedance Toolbox

## Installation
**NB:** Though most of this PyWIT's functionality is available independently of any non-dependency package,
the installation of IW2D (https://gitlab.cern.ch/IRIS/IW2D) is *strongly recommended* in order to have full access to 
all of the features of PyWIT. For this reason, this installation guide will include the installation process required by 
IW2D. If you already have IW2D downloaded and installed on your system, you can skip directly to step 3. Please note that IW2D (at the time of writing) is only available on Linux systems.

1. Download the IW2D repository to your system by cloning the git repository. This can be done by executing the command
   
    `git clone https://gitlab.cern.ch/IRIS/IW2D` 
   
    in a suitable directory. You may be prompted to enter your CERN log-in credentials for this step.


2. Perform the installation procedure for IW2D. This is explained in detail in the `README.md` file in the IW2D repository, but is summarized here:
   1. Install the prerequisite libraries "MPFR" and "GSL" by running:  
   `sudo apt-get install libgsl-dev libmpfr-dev`  
   or alternatively, if you have anaconda/miniconda installed:  
   `conda install -c conda-forge gsl mpfr`
   2. Compile the C++ code by running:
      ```
      cd IW2D/cpp
      cp ./Makefile_system_GMP_MPFR Makefile
      make
      ```


3. In a new directory, where you want your PyWIT directory to be placed, Download the PyWIT repository to your system by 
   cloning the git repository. This can be done by executing the command
   
   `git clone https://gitlab.cern.ch/IRIS/pywit`.

   You may be prompted to enter your CERN log-in credentials for this step.


4. Install PyWIT using pip by navigating to your pywit directory executing the command

   `pip install .`.


5. Navigate to the directory `IW2D/IW2D/cpp` in the IW2D repository. Copy the following four
files from this folder:
   * `flatchamber.x`
   * `roundchamber.x`
   * `wake_flatchamber.x`
   * `wake_roundchamber.x`
   

6. In your home directory, there will have appeared a new directory called 'pywit'. Navigate into this folder, then
into 'IW2D', then into 'bin'. Paste the four files copied in step 5 into this folder.


You should now be able to use PyWIT with your Python system interpreter.
