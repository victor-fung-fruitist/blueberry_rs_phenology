from pathlib import Path
import git


# base folders
REPO_ROOT: str = git.Repo(
    ".",
    search_parent_directories=True,
).working_tree_dir

# folder structure
FOLDER_STRUCTURE = {
    "data": {
        "01_raw": None,
        "02_intermediate": None,
        "03_primary": None,
        "04_feature": None,
        "05_model_input": None,
        "06_models": None,
        "07_model_output": None,
        "08_reporting": None,
    }
}


def create_folder_structure(
    root_path: str,
    structure: dict,
):
    """
    Creates a folder structure starting from a root path using pathlib.

    Parameters:
    - root_path: The base directory in which the folders will be created, as a Path object or a string.
    - structure: A dictionary where each key is a folder name and each value is
                 either None or another dictionary defining a more nested structure.
    """
    root = Path(root_path)
    for folder, substructure in structure.items():
        # Create the path to the current folder
        path = root / folder

        # Make the directory if it does not exist
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Create a .gitkeep file in the folder
        Path(path / ".gitkeep").touch()

        # If there are subdirectories, recurse into those
        if substructure is not None:
            create_folder_structure(path, substructure)


# Create the folder structure
create_folder_structure(
    root_path=REPO_ROOT,
    structure=FOLDER_STRUCTURE,
)
