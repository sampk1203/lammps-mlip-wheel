#!/usr/bin/env python3
"""
run_lammps.py

Thin CLI wrapper around the LAMMPS Python library that mimics the native
`lmp` binary: takes a standard LAMMPS input script and runs it, either
serially or under MPI via `mpirun`. Built on top of the lammps-mlip-wheel
package (https://github.com/sampk1203/lammps-mlip-wheel).

Usage (same flags as the real `lmp` binary):
    python run_lammps.py -in in.melt
    mpirun -np 4 python run_lammps.py -in in.melt
    mpirun -np 4 python run_lammps.py -in in.melt -var temp 300 -var press 1.0
    python run_lammps.py -in in.melt -log run.log -screen none

Also importable for post-processing, without tearing down the LAMMPS
instance immediately after the run:

    from run_lammps import run_lammps

    l = run_lammps("in.melt", close=False)
    # l is still open here -- pull data out before closing
    natoms = l.get_natoms()
    pe = l.get_thermo("pe")
    l.close()
"""

import argparse
import sys
import traceback

from mpi4py import MPI
from lammps import lammps


def build_lammps_args(log, screen, extra_lmp_args):
    """Translate our CLI options into LAMMPS's own -log/-screen/-var style argv.

    Note: do NOT prepend an argv[0]/executable-name placeholder here -- the
    lammps Python constructor already adds that itself. Passing one manually
    causes LAMMPS to see a duplicate, unrecognized first argument.
    """
    args = []
    if log is not None:
        args += ["-log", log]
    if screen is not None:
        args += ["-screen", screen]
    args += extra_lmp_args
    return args


def run_lammps(infile, log=None, screen=None, variables=None, close=True):
    """
    Run a LAMMPS input script through the Python library interface.

    Parameters
    ----------
    infile : str
        Path to a LAMMPS input script (what you'd normally pass to `lmp -in`).
    log : str, optional
        Path for LAMMPS's own log file. Pass "none" to suppress logging.
        Defaults to LAMMPS's own default (log.lammps) if not given.
    screen : str, optional
        Where to send LAMMPS's screen output. Pass "none" to suppress.
    variables : list[list[str, str]], optional
        (name, value) pairs, equivalent to `-var name value` on the lmp
        command line. Useful for parameterized input scripts.
    close : bool, default True
        If True, close the LAMMPS instance before returning -- normal
        run-and-done use. If False, the caller owns the instance and must
        call l.close() themselves -- use this when you want to pull data
        out for post-processing (l.get_thermo(), l.numpy.extract_atom(...),
        etc.) before the instance is torn down.

    Returns
    -------
    lammps.lammps
        The LAMMPS instance. Still open if close=False.
    """
    extra_lmp_args = []
    if variables:
        for name, value in variables:
            extra_lmp_args += ["-var", name, value]

    lmp_args = build_lammps_args(log, screen, extra_lmp_args)
    l = lammps(cmdargs=lmp_args)

    l.file(infile)

    if close:
        l.close()

    return l


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a LAMMPS input script via the Python library interface."
    )
    parser.add_argument(
        "-in", "--input", dest="infile", required=True,
        help="Path to the LAMMPS input script (same as lmp -in <file>).",
    )
    parser.add_argument(
        "-log", dest="log", default=None,
        help="Path for LAMMPS's log file (same as lmp -log <file>). "
             "Pass 'none' to suppress logging.",
    )
    parser.add_argument(
        "-screen", dest="screen", default=None,
        help="Screen output target (same as lmp -screen <file|none>).",
    )
    parser.add_argument(
        "-var", dest="variables", nargs=2, action="append", metavar=("NAME", "VALUE"),
        help="Define a LAMMPS index/string variable, same as lmp -var NAME VALUE. "
             "Repeatable for multiple variables.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rank = MPI.COMM_WORLD.Get_rank()

    try:
        run_lammps(
            args.infile,
            log=args.log,
            screen=args.screen,
            variables=args.variables,
            close=True,
        )
    except Exception:
        # Still finalize MPI cleanly even if LAMMPS itself raised, so the
        # job doesn't hang waiting on ranks that never get here.
        if rank == 0:
            traceback.print_exc()
        MPI.Finalize()
        sys.exit(1)

    MPI.Finalize()


if __name__ == "__main__":
    main()
