# lammps-mlip-wheel

Portable `manylinux_2_28` wheel of LAMMPS (`2026.2.11`) with KIM, MDI, PLUMED,
ML-IAP, netCDF, COLVARS, MPI, and ASE-compatible MLIP support via
`gnnp_driver`. Excludes ADIOS, GPU, KOKKOS, MBX, SCAFACOS, VTK.

MPI, libcurl, and libldap are **not** bundled in the wheel — they must be
present on the target system already (see below).

## Requirements

- **Python 3.12 only.** The wheel is tagged `cp312` and will not import under
  3.11, 3.13, or any other minor version. Confirm before installing:
  ```bash
  python3 --version   # must print 3.12.x
  ```
- Linux, glibc-compatible with `manylinux_2_28` (roughly RHEL 8 / Ubuntu 20.04+).

## Install runtime dependencies

**Debian / Ubuntu / Pop!_OS:**
```bash
sudo apt update
sudo apt install -y \
    libopenmpi3 openmpi-bin \
    libcurl4 \
    libldap-2.5-0 \
    ffmpeg
```
- `libopenmpi3` — runtime MPI library, matches the v40-series ABI the wheel links against.
- `openmpi-bin` — provides `mpirun`, needed for multi-rank use.
- `libcurl4` / `libldap-2.5-0` — excluded from the wheel to prevent a shutdown-time segfault; must be present on the system.
- `ffmpeg` — optional, only needed for `dump movie`.

**RHEL / AlmaLinux / Rocky Linux** (package names correct for the family, not independently verified end-to-end):
```bash
sudo dnf install -y openmpi libcurl openldap

# ffmpeg from RPM Fusion (not in base repos or EPEL):
sudo dnf install -y https://download1.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm
sudo dnf install -y ffmpeg
```

## Install the wheel

```bash
pip install "https://github.com/sampk1203/lammps-mlip-wheel/releases/download/v2026.2.11/lammps-2026.2.11-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
```
or with `uv`:
```bash
uv add "https://github.com/sampk1203/lammps-mlip-wheel/releases/download/v2026.2.11/lammps-2026.2.11-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
```

## KIM model data

The wheel ships the KIM API library only, not model data:
```bash
kim-api-collections-management install user <model-name>
```

## Verify

```bash
# Basic import and package list
python -c "from lammps import lammps; l = lammps(); print(l.installed_packages)"
echo "exit: $?"

# MPI (multi-rank)
mpirun -np 2 python -c \
    "from lammps import lammps; l = lammps(); \
     l.commands_string('units lj\natom_style atomic\nlattice fcc 0.8442\n\
region box block 0 4 0 4 0 4\ncreate_box 1 box\ncreate_atoms 1 box\n\
mass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0 2.5\nrun 10'); \
     print('mpi ok')" -quiet
echo "mpi exit: $?"
```
Both should exit `0` and print the installed-packages list / `mpi ok`.
