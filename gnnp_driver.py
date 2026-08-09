"""
Copyright (c) 2025, AdvanceSoft Corp.

This source code is licensed under the GNU General Public License Version 2
found in the LICENSE file in the root directory of this source tree.
"""

from ase import Atoms
from ase.calculators.mixing import SumCalculator

import os
import torch
import warnings

_USING_TORCH_DFTD3 = True

# Set by gnnp_initialize() when gnnp_type == "orbmol"; consumed by
# gnnp_get_energy_forces_stress() to populate atoms.info["charge"/"spin"],
# which newer orb_models versions require instead of calculator kwargs.
_orbGlobalChargeSpin = None

def gnnp_initialize(gnnp_type, model_name = None, as_path = False, dftd3 = False, gpu = True, **kwargs):
    """
    Initialize GNNP.
    Args:
        gnnp_type (str): type of GNNP. -> {matgl|chgnet|mace|mace-off|orb|mattersim|fairchem|sevennet}
        model_name (str): name of model for GNNP.
        as_path (bool): if true, model_name is path of model file. this is only for chgnet/orb/fairchem.
        dftd3 (bool): to add correction of DFT-D3.
        gpu (bool): using GPU, if possible.
        **kwargs: extra per-model arguments forwarded from trailing `key=value`
            tokens on the LAMMPS `pair_coeff` line (e.g. charge, spin). Only
            the branches that need them consume specific keys via
            `kwargs.pop(...)`; anything left over after dispatch is reported
            via `warnings.warn(...)` and otherwise ignored, it is not an error.
    Returns:
        cutoff (float): cutoff radius.
        with_stress (int): to calculate stress, or not.
    """

    # Check gpu
    gpu    = (gpu and torch.cuda.is_available())
    device = "cpu" if gpu else "cpu"

    # Create Calculator of GNNP, that is pre-trained
    global myCalculator

    myCalculator = None
    cutoff       = -1.0

    # Reset any stale state from a previous orbmol init; only the "orbmol"
    # branch below will set this again if applicable.
    global _orbGlobalChargeSpin
    _orbGlobalChargeSpin = None

    if gnnp_type is None:
        raise ValueError("gnnp_type is not defined.")

    gnnp_type = gnnp_type.lower()

    if gnnp_type == "matgl":
        # MatGL
        import matgl
        from matgl.ext.ase import PESCalculator

        torch.set_default_device(device)

        if model_name is not None:
            myPotential = matgl.load_model(model_name)
        else:
            myPotential = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        myPotential.to(device)

        myCalculator = PESCalculator(
            potential      = myPotential,
            compute_stress = True,
            stress_unit    = "eV/A3",
            stress_weight  = 1.0
        )

        cutoff = myPotential.model.cutoff

    elif gnnp_type == "chgnet":
        # CHGNet
        from chgnet.model import CHGNet, CHGNetCalculator

        if model_name is None:
            myCHGNet = CHGNet.load(use_device = device)
        elif not as_path:
            myCHGNet = CHGNet.load(use_device = device, model_name = model_name)
        else:
            myCHGNet = CHGNet.from_file(model_name)

        myCalculator = CHGNetCalculator(
            model      = myCHGNet,
            use_device = device
        )

        ratom  = float(myCHGNet.graph_converter.atom_graph_cutoff)
        rbond  = float(myCHGNet.graph_converter.bond_graph_cutoff)
        cutoff = max(ratom, rbond)

    elif gnnp_type == "sevennet":
        from sevenn.calculator import SevenNetD3Calculator

        if model_name is None:
            model_name = "7net-0"

        # Build args depending on model
        calc_kwargs = {"model": model_name, "device": device}

        if model_name == "7net-mf-ompa":
            # This model requires modal argument
            calc_kwargs["modal"] = "mpa"  # or "omat24", depending on what you want

        myCalculator = SevenNetD3Calculator(**calc_kwargs)

        cutoff = myCalculator.cutoff if hasattr(myCalculator, "cutoff") else 4.0


    elif gnnp_type == "mace":
        # MACE
        from mace.calculators import mace_mp

        if model_name is None:
            model = None

        elif model_name.startswith("mace-osaka24"):
            base_path  = os.path.dirname (os.path.abspath(__file__))
            model_dir  = os.path.normpath(os.path.join(base_path, "mace-osaka24"))
            model_path = os.path.normpath(os.path.join(model_dir, model_name))

            if not model_path.endswith(".model"):
                model_path += ".model"

            model = model_path

        else:
            model = model_name

        myCalculator = mace_mp(
            model         = model,
            device        = device,
            dispersion    = dftd3,
            damping       = "zero",
            dispersion_xc = "pbe"
        )

        if dftd3:
            dftd3 = False

        if isinstance(myCalculator, SumCalculator):
            cutoff = myCalculator.mixer.calcs[0].r_max
        else:
            cutoff = myCalculator.r_max

    elif gnnp_type == "mace-off":
        # MACE-OFF
        from mace.calculators import mace_off

        myCalculator = mace_off(
            model  = model_name,
            device = device
        )

        cutoff = myCalculator.r_max

    elif gnnp_type in ("orb", "orbmol"):
        # Orbital Materials.
        #   "orb"    -> orb-v3 structural models (periodic solids, e.g. omat/mpa;
        #               has a stress head, no charge/spin conditioning).
        #   "orbmol" -> OrbMol models (molecular, OMol-trained; charge/spin
        #               conditioned via atoms.info; orbmol-v2 has NO stress head).
        # As of orb_models >= 0.6, ORBCalculator moved under
        # orb_models.forcefield.inference.calculator, and every pretrained.*
        # loader now returns (model, atoms_adapter) instead of a bare model;
        # ORBCalculator requires that atoms_adapter.
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator

        is_molecular = (gnnp_type == "orbmol")
        default_key  = "orbmol-v2" if is_molecular else "orb-v3-conservative-inf-omat"

        # Optional explicit override of which pretrained loader to use when
        # as_path=True, since a fine-tuned checkpoint's weights must be loaded
        # into the matching architecture (v2 vs v3, conservative vs direct,
        # structural vs OrbMol) -- this can't be inferred from the file alone.
        # e.g. pair_coeff ... orb my_checkpoint.ckpt arch=orb-v3-direct-20-omat
        arch = kwargs.pop("arch", None)

        if as_path:
            loader_key = (arch if arch is not None else default_key).replace("_", "-")
            if loader_key not in pretrained.ORB_PRETRAINED_MODELS:
                raise ValueError(
                    "Unknown ORB architecture '" + loader_key + "' for as_path; "
                    "pass a valid key via arch=... (see pretrained.ORB_PRETRAINED_MODELS)."
                )
            model_func  = pretrained.ORB_PRETRAINED_MODELS[loader_key]
            orb_model, atoms_adapter = model_func(weights_path = model_name, device = device)

        else:
            loader_key = (model_name if model_name is not None else default_key).replace("_", "-")
            if loader_key not in pretrained.ORB_PRETRAINED_MODELS:
                raise ValueError(
                    "Unknown ORB model '" + str(model_name) + "'; "
                    "see pretrained.ORB_PRETRAINED_MODELS for valid keys."
                )
            model_func  = pretrained.ORB_PRETRAINED_MODELS[loader_key]
            orb_model, atoms_adapter = model_func(device = device)

        myCalculator = ORBCalculator(
            orb_model,
            atoms_adapter = atoms_adapter,
            device        = device
        )

        cutoff = float(atoms_adapter.radius)

        # Avoid double-counting dispersion if the chosen model already
        # bundles a D3/D4 correction (e.g. orb-d3-v2, separate-d3-3layer).
        if dftd3 and ("d3" in loader_key or "d4" in loader_key):
            dftd3 = False

        if is_molecular:
            # charge/spin are per-system, forwarded from pair_coeff, and must
            # be written into atoms.info at calculate time (see
            # gnnp_get_energy_forces_stress below) -- OrbMol raises if either
            # is missing.
            _orbGlobalChargeSpin = (
                float(kwargs.pop("charge", 0)),
                float(kwargs.pop("spin", 1))
            )

    elif gnnp_type == "mattersim":
        # MatterSim
        from mattersim.forcefield import MatterSimCalculator

        myCalculator = MatterSimCalculator(
            load_path      = model_name,
            compute_stress = True,
            device         = device
        )

        cutoff = myCalculator.potential.model.model_args.get("cutoff", 5.0)

    elif gnnp_type == "fairchem":
        # FAIR-Chem
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator

        # Extra per-model args (e.g. charge, spin) forwarded from pair_coeff.
        extraArgs = {}
        if "charge" in kwargs:
            extraArgs["charge"] = kwargs.pop("charge")
        if "spin" in kwargs:
            extraArgs["spin"] = kwargs.pop("spin")

        if as_path:
            myCalculator = OCPCalculator(
                checkpoint_path = model_name,
                cpu             = not gpu,
                **extraArgs
            )

        else:
            OMAT_CHECKPTS = {
                "EquiformerV2-31M-OMat"          : "eqV2_31M_omat.pt",
                "EquiformerV2-86M-OMat"          : "eqV2_86M_omat.pt",
                "EquiformerV2-153M-OMat"         : "eqV2_153M_omat.pt",
                "EquiformerV2-31M-MP"            : "eqV2_31M_mp.pt",
                "EquiformerV2-31M-DeNS-MP"       : "eqV2_dens_31M_mp.pt",
                "EquiformerV2-86M-DeNS-MP"       : "eqV2_dens_86M_mp.pt",
                "EquiformerV2-153M-DeNS-MP"      : "eqV2_dens_153M_mp.pt",
                "EquiformerV2-31M-OMat-Alex-MP"  : "eqV2_31M_omat_mp_salex.pt",
                "EquiformerV2-86M-OMat-Alex-MP"  : "eqV2_86M_omat_mp_salex.pt",
                "EquiformerV2-153M-OMat-Alex-MP" : "eqV2_153M_omat_mp_salex.pt",
            }

            if model_name is not None:
                checkpt_name = OMAT_CHECKPTS.get(model_name);
            else:
                checkpt_name = OMAT_CHECKPTS.get("EquiformerV2-31M-OMat");

            if checkpt_name is not None:
                base_path   = os.path.dirname (os.path.abspath(__file__))
                checkpt_dir = os.path.normpath(os.path.join(base_path, "fairchem-omat24"))
                model_path  = os.path.normpath(os.path.join(checkpt_dir, checkpt_name))

                myCalculator = OCPCalculator(
                    checkpoint_path = model_path,
                    cpu             = not gpu,
                    **extraArgs
                )

            else:
                #base_path   = os.path.dirname (os.path.abspath(__file__))
                base_path   = os.path.expanduser("~")
                checkpt_dir = os.path.normpath(os.path.join(base_path, ".fairchem"))

                myCalculator = OCPCalculator(
                    model_name  = model_name,
                    local_cache = checkpt_dir,
                    cpu         = not gpu,
                    **extraArgs
                )

        cutoff = myCalculator.config["model"].get("max_radius", 8.0)

    else:
        raise ValueError("gnnp_type is incorrect: " + gnnp_type)

    # Any trailing pair_coeff key=value tokens not consumed by the matched
    # branch above are reported, but do not abort initialization.
    if kwargs:
        warnings.warn(
            "Unused pair_coeff arguments for " + gnnp_type + ": " + str(list(kwargs.keys()))
        )

    if "stress" in myCalculator.implemented_properties:
        with_stress = 1
    else:
        with_stress = 0

    # Add DFT-D3 to calculator without three-body term
    global gnnpCalculator
    global dftd3Calculator

    gnnpCalculator  = myCalculator
    dftd3Calculator = None

    if dftd3:
        if _USING_TORCH_DFTD3:
            from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

            dftd3Calculator = TorchDFTD3Calculator(
                xc      = "pbe",
                damping = "zero",
                abc     = False,
                device  = device
            )

        else:
            from dftd3.ase import DFTD3

            dftd3Calculator = DFTD3(
                method  = "PBE",
                damping = "d3zero",
                s9      = 0.0
            )

        myCalculator = SumCalculator([gnnpCalculator, dftd3Calculator])

    # Atoms object of ASE, that is empty here
    global myAtoms

    myAtoms = None

    return (cutoff, with_stress)

def gnnp_get_energy_forces_stress(cell, atomic_numbers, positions, with_stress = True):
    """
    Predict total energy, atomic forces and stress w/ pre-trained GNNP.
    Args:
        cell: lattice vectors in angstroms.
        atomic_numbers: atomic numbers for all atoms.
        positions: xyz coordinates for all atoms in angstroms.
        with_stress: to return stress, if True.
    Returns:
        energy:  total energy.
        forcces: atomic forces.
        stress:  stress tensor (Voigt order).
    """

    # Initialize Atoms
    global myAtoms
    global myCalculator

    if myAtoms is not None and len(myAtoms.numbers) != len(atomic_numbers):
        myAtoms = None

    if myAtoms is None:
        myAtoms = Atoms(
            numbers   = atomic_numbers,
            positions = positions,
            cell      = cell,
            pbc       = [True, True, True]
        )

        myAtoms.calc = myCalculator

    else:
        myAtoms.set_cell(cell)
        myAtoms.set_atomic_numbers(atomic_numbers)
        myAtoms.set_positions(positions)

    # OrbMol models require charge/spin on atoms.info at calculate time.
    global _orbGlobalChargeSpin
    if _orbGlobalChargeSpin is not None:
        myAtoms.info["charge"] = _orbGlobalChargeSpin[0]
        myAtoms.info["spin"]   = _orbGlobalChargeSpin[1]

    # Predicting energy, forces and stress
    energy = myAtoms.get_potential_energy()
    if not isinstance(energy, float):
        energy = energy.item()

    forces = myAtoms.get_forces().tolist()

    if not with_stress:
        return energy, forces

    global gnnpCalculator
    global dftd3Calculator

    if dftd3Calculator is None:
        stress = myAtoms.get_stress().tolist()
    else:
        # to avoid the bug of SumCalculator
        myAtoms.calc = gnnpCalculator
        stress1 = myAtoms.get_stress()

        myAtoms.calc = dftd3Calculator
        stress2 = myAtoms.get_stress()

        stress = stress1 + stress2
        stress = stress.tolist()

        myAtoms.calc = myCalculator

    return energy, forces, stress
