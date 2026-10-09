# lammps-mlip-wheel

Portable `manylinux_2_28` wheel of LAMMPS with KIM, MDI, PLUMED,
ML-IAP, netCDF, COLVARS, MPI, and ASE-compatible MLIP support via
`gnnp_driver`. Excludes ADIOS, KOKKOS, MBX,FENIX, QMMM-XTB, SCAFACOS, VTK.

MPI, libcurl, and libldap are **not** bundled in the wheel — they must be
present on the target system already (see below).

## Install runtime dependencies

**Debian / Ubuntu / Pop!_OS:**
```bash
sudo apt update
sudo apt install -y \
libfftw3-dev \
libeigen3-dev \
libcurl4-openssl-dev \
zlib1g-dev \
libjpeg-dev \
libpng-dev \
libzstd-dev \
libnetcdf-dev \
libpnetcdf-dev \
libgsl-dev \
libopenblas-dev \
libhdf5-serial-dev \
libkim-api-dev \
cython3 \
ffmpeg \
gcc g++ gfortran \
libopenmpi-dev openmpi-bin openmpi-common \
python3 python3-dev python3-pip python3-venv python3-full
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
sudo dnf install -y ffmpeg {and other packages which are dnf variants from the apt list}
```

## Install the wheel

```bash
pip install "https://github.com/sampk1203/lammps-mlip-wheel/releases/download/v2026.2.11/lammps-{relavant version from releases}-.whl"
```
or with `uv`:
```bash
uv add "https://github.com/sampk1203/lammps-mlip-wheel/releases/download/v2026.2.11/lammps-{relavant version from releases}-.whl"
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
