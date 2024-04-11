"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = ksz.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import sys

from scipy.interpolate import CubicSpline
from ksz import __version__

__author__ = "Lisa McBride"
__copyright__ = "Lisa McBride"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from ksz.skeleton import fib`,
# when using this Python module as a library.


def tau(n):
    """Calculates tau given an ionisation history xe(z)

    Args:
        z  (array_like): redshift
        xe (array_like): ionisation history

    Returns:
        z  (array_like): redshift
        tau (array_like): tau (cumulative integral)
    """
    z_unity = 5.0 # redshift at which hydrogen is has an ionisation fraction x_HII=1
    z_HeII = 3.5  # redshift at which helium doubly ionises

    if z.min() <= z_unity:
        z_extra = np.linspace(0.0, z.min(), endpoint=False)
    else:
        z_extra = np.linspace(0.0, z_unity)

    xe_lowz = np.ones_like(z_extra)

    spl = CubicSpline(x, y)

    return z, tau

def xe_allz(z, xe):
    """Calculates tau given an ionisation history xe(z)

    Args:
        z  (array_like): redshift
        xe (array_like): ionisation history

    Returns:
        z  (array_like): redshift
        xe (array_like): extrapolated up to z=0
    """
    z_unity = 5.0 # redshift at which hydrogen is has an ionisation fraction x_HII=1
    z_HeII = 3.5  # redshift at which helium doubly ionises

    if z.min() <= z_unity:
        z_extra = np.linspace(0.0, z.min(), endpoint=False)
    else:
        z_extra = np.linspace(0.0, z_unity)

    xe_lowz = np.ones_like(z_extra)
    z2interpl = np.concatenate(z_extra, z)
    xe2interpl = np.concatenate(xe_lowz, xe)

    xe_all = CubicSpline(z_interpl, xe_interpl)

    z_all = np.linspace(0.0, z_max(), 1000)

    return z_all, xe_all(z_all)

# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration")
    parser.add_argument(
        "--version",
        action="version",
        version=f"kSZ {__version__}",
    )
    parser.add_argument(dest="n", help="n-th Fibonacci number", type=int, metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    print(f"The {args.n}-th Fibonacci number is {fib(args.n)}")
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m ksz.skeleton 42
    #
    run()
