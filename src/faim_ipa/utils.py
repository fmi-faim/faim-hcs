import logging
import pathlib
from datetime import datetime
import os.path
from pathlib import Path

import pydantic
from pydantic import BaseModel


def wavelength_to_rgb(wavelength, gamma=0.8):
    """This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).
    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html

    Obtained from https://gist.github.com/error454/65d7f392e1acd4a782fc
    """

    wavelength = float(wavelength)
    if 380 <= wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif 510 <= wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif 645 <= wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return int(R), int(G), int(B)


def rgb_to_hex(r, g, b):
    """Convert RGB values into hex number."""
    return f"{r:02x}{g:02x}{b:02x}"


def create_logger(name: str) -> logging.Logger:
    """
    Create logger which logs to <timestamp>-<name>.log inside the current
    working directory.

    Parameters
    ----------
    name
        Name of the logger instance.
    """
    logger = logging.Logger(name.capitalize())
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    handler = logging.FileHandler(f"{now}-{name}.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_git_root() -> Path:
    """
    Recursively search for the directory containing the .git folder.

    Returns
    -------
    Path
        Path to the root of the git repository.
    """
    parent_dir = Path(__file__).parent
    while not (parent_dir / ".git").exists():
        parent_dir = parent_dir.parent

    return parent_dir


def resolve_with_git_root(relative_path: Path) -> Path:
    """
    Takes a relative path and resolves it relative to the git_root directory.

    Parameters
    ----------
    relative_path
        Path relative to the git root.

    Returns
    -------
    Path
        Absolute path to the file.
    """
    git_root = get_git_root()
    return (git_root / relative_path).resolve()


def make_relative_to_git_root(path: Path) -> Path:
    """
    Convert an absolute path to a path relative to the git_root directory.

    Parameters
    ----------
    path
        Absolute path to a file.

    Returns
    -------
    Path
        Path relative to the git root.
    """
    git_root = get_git_root()
    try:
        # requires Python >= 3.12
        return path.relative_to(git_root, walk_up=True)
    except (ValueError, TypeError):
        # fallback for Python < 3.12
        return Path(os.path.relpath(path, git_root))


class IPAConfig(BaseModel):

    def make_paths_absolute(self):
        """
        Convert all `pathlib.Path` fields to absolute paths.

        The paths are assumed to be relative to a git-root directory somewhere
        in the parent directories of the class implementing `IPAConfig`.
        """
        if pydantic.__version__.startswith("2"):
            fields = self.model_fields_set
        else:
            fields = self.__fields_set__

        for f in fields:
            attr = getattr(self, f)
            if isinstance(attr, pathlib.Path) and not attr.is_absolute():
                setattr(self, f, resolve_with_git_root(attr))

    def make_paths_relative(self):
        """
        Convert all `pathlib.Path` fields to relative paths.

        The resulting paths will be relative to the git-root directory
        somewhere in the parent directories of the class implementing
        `IPAConfig`.
        """
        if pydantic.__version__.startswith("2"):
            fields = self.model_fields_set
        else:
            fields = self.__fields_set__

        for f in fields:
            attr = getattr(self, f)
            if isinstance(attr, pathlib.Path) and attr.is_absolute():
                setattr(self, f, make_relative_to_git_root(attr))
