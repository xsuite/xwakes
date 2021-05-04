from setuptools import setup, find_packages
from setuptools.command.install import install
from os import makedirs
import pickle
import pathlib


def _initialize_pywib_directory() -> None:
    home_path = pathlib.Path.home()
    paths = [home_path.joinpath('pywib').joinpath(ext) for ext in ('config', 'IW2D/bin', 'IW2D/projects')]
    for path in paths:
        makedirs(path, exist_ok=True)
    with open(pathlib.Path(paths[0]).joinpath('iw2d_settings.yaml'), 'w') as file:
        file.write(f'binary_directory: {paths[1]}\n'
                   f'project_directory: {paths[2]}')

    filenames = ('component', 'element', 'iw2d_inputs')
    for filename in filenames:
        open(paths[0].joinpath(f"{filename}.yaml"), 'w').close()

    with open(paths[2].joinpath('hashmap.pickle'), 'wb') as handle:
        pickle.dump(dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        _initialize_pywib_directory()


setup(
    name='pywib',
    version='1.0.0',
    packages=find_packages(),
    url='https://gitlab.cern.ch/mrognlie/pywib',
    license='MIT',
    author='Markus Kongstein Rognlien',
    author_email='marro98@gmail.com',
    description='Python Wake and Impedance Budget',
    cmdclass={'install': PostInstallCommand}
)
