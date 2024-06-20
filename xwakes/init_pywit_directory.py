import pathlib
import pickle
import os

def initialize_pywit_directory() -> None:
    home_path = pathlib.Path.home()
    paths = [home_path.joinpath('pywit').joinpath(ext) for ext in ('config', 'IW2D/bin', 'IW2D/projects')]
    for path in paths:
        os.makedirs(path, exist_ok=True)
    with open(pathlib.Path(paths[0]).joinpath('iw2d_settings.yaml'), 'w') as file:
        file.write(f'binary_directory: {paths[1]}\n'
                   f'project_directory: {paths[2]}')

    filenames = ('component', 'element', 'iw2d_inputs')
    for filename in filenames:
        open(paths[0].joinpath(f"{filename}.yaml"), 'w').close()

    with open(paths[2].joinpath('hashmap.pickle'), 'wb') as handle:
        pickle.dump(dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
