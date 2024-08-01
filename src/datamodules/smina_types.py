import argparse
import multiprocessing as mp
import urllib
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd

# from openbabel import openbabel
# from openbabel import pybel
import rdkit
from rdkit import Chem
import os
from dataclasses import dataclass
from src.datamodules.constants import ADNAMES, ATOM_TYPES


@dataclass
class SminaAtomType:
    name: str
    radius: float
    depth: float
    solvation: float
    volume: float
    covalent_radius: float
    xs_radius: float
    xs_hydrophobe: int
    xs_donor: int
    xs_acceptor: int
    ad_heteroatom: int
    adname: str = None

    def __str__(self):
        return f"{self.name} {self.radius} {self.depth} {self.solvation} {self.volume} {self.covalent_radius} {self.xs_radius} {self.xs_hydrophobe} {self.xs_donor} {self.xs_acceptor} {self.ad_heteroatom}"


class StructuralFileParser:
    def __init__(self):
        self.atom_type_data = []
        self.type_map = {}
        self.rdkit_funcs = {
            ".pdb": Chem.MolFromPDBFile,
            ".sdf": Chem.MolFromMolFile,
            ".mol2": Chem.MolFromMol2File,
            ".mol": Chem.MolFromMolFile,
        }
        lines = ATOM_TYPES.split("\n")
        adnames = ADNAMES
        for i in lines[1:]:
            atom = SminaAtomType(*i.split())
            atom.adname = adnames[atom.name]
            self.atom_type_data.append(atom)
            self.type_map[atom.name] = atom
        self.atom_types = [info.name for info in self.atom_type_data]
        self.type_map = self.get_types_map()
        self.n_features = len(set(self.type_map.values())) + 1

    def read_file(self, infile):
        ext = Path(infile).suffix
        return self.rdkit_funcs[ext](str(infile))

    def get_types_map(self):
        types = [
            ["AliphaticCarbonXSHydrophobe"],
            ["AliphaticCarbonXSNonHydrophobe"],
            ["AromaticCarbonXSHydrophobe"],
            ["AromaticCarbonXSNonHydrophobe"],
            ["Nitrogen", "NitrogenXSAcceptor"],
            ["NitrogenXSDonor", "NitrogenXSDonorAcceptor"],
            ["Oxygen", "OxygenXSAcceptor"],
            ["OxygenXSDonor", "OxygenXSDonorAcceptor"],
            ["Sulfur", "SulfurAcceptor", "Selenium"],
            ["Phosphorus"],  # == 9
        ]
        out_dict = defaultdict(lambda: len(types))
        for i, element_name in enumerate(self.atom_types):
            for types_list in types:
                if element_name in types_list:
                    out_dict[i] = types.index(types_list)
                    break
        return out_dict

    @staticmethod
    def adjust_smina_type(t, h_bonded, hetero_bonded):
        """Original author: Constantin schneider"""
        if t in (
            "AliphaticCarbonXSNonHydrophobe",
            "AliphaticCarbonXSHydrophobe",
        ):  # C_C_C_P,
            if hetero_bonded:
                return "AliphaticCarbonXSNonHydrophobe"
            else:
                return "AliphaticCarbonXSHydrophobe"
        elif t in (
            "AromaticCarbonXSNonHydrophobe",
            "AromaticCarbonXSHydrophobe",
        ):  # C_A_C_P,
            if hetero_bonded:
                return "AromaticCarbonXSNonHydrophobe"
            else:
                return "AromaticCarbonXSHydrophobe"
        elif t in ("Nitrogen", "NitogenXSDonor"):
            # N_N_N_P, no hydrogen bonding
            if h_bonded:
                return "NitrogenXSDonor"
            else:
                return "Nitrogen"
        elif t in ("NitrogenXSAcceptor", "NitrogenXSDonorAcceptor"):
            # N_NA_N_A, also considered an acceptor by autodock
            if h_bonded:
                return "NitrogenXSDonorAcceptor"
            else:
                return "NitrogenXSAcceptor"
        elif t in ("Oxygen" or t == "OxygenXSDonor"):  # O_O_O_P,
            if h_bonded:
                return "OxygenXSDonor"
            else:
                return "Oxygen"
        elif t in ("OxygenXSAcceptor" or t == "OxygenXSDonorAcceptor"):
            # O_OA_O_A, also an autodock acceptor
            if h_bonded:
                return "OxygenXSDonorAcceptor"
            else:
                return "OxygenXSAcceptor"
        else:
            return t

    def get_is_hbond_acceptor(self, rdkit_atom):
        is_neg = rdkit_atom.GetFormalCharge() < 0
        non_hydrogen_neighbours = [
            n for n in rdkit_atom.GetNeighbors() if n.GetAtomicNum() != 1
        ]
        max_valence = {7: 3, 16: 6}
        is_potentially_unsaturated = (
            len(non_hydrogen_neighbours) < max_valence[rdkit_atom.GetAtomicNum()]
        )
        return is_neg or is_potentially_unsaturated

    def rdkitatom_to_smina_type(self, rdkit_atom):
        atomic_number = rdkit_atom.GetAtomicNum()
        num_to_name = {1: "HD", 6: "A", 7: "NA", 8: "OA", 16: "SA"}

        # # Default fn returns True, otherwise inspect atom properties
        # condition_fns = defaultdict(lambda: lambda: True)
        # condition_fns.update(
        #     {
        #         6: rdkit_atom.GetIsAromatic,
        #         7: self.get_is_hbond_acceptor(rdkit_atom),
        #         16: self.get_is_hbond_acceptor(rdkit_atom),
        #     }
        # )

        # Get symbol
        ename = rdkit_atom.GetSymbol()

        # Do we need to adjust symbol?
        if atomic_number == 6 and rdkit_atom.GetIsAromatic():
            ename = num_to_name.get(atomic_number, ename)
        elif atomic_number in (7, 8, 16):
            ename = num_to_name.get(atomic_number, ename)

        atype = self.string_to_smina_type(ename)

        h_bonded = False
        hetero_bonded = False
        for neighbour in rdkit_atom.GetNeighbors():
            if neighbour.GetAtomicNum() == 1:
                h_bonded = True
            elif neighbour.GetAtomicNum() != 6:
                hetero_bonded = True

        return self.adjust_smina_type(atype, h_bonded, hetero_bonded)

    def string_to_smina_type(self, string: str):
        """Convert string type to smina type.

        Original author: Constantin schneider

        Args:
            string (str): string type
        Returns:
            string: smina type
        """
        if len(string) <= 2:
            for type_info in self.atom_type_data:
                # convert ad names to smina types
                if string == type_info.adname:
                    return type_info.name
            # generic metal
            # if string in self.non_ad_metal_names:
            #     return "GenericMetal"
            # # if nothing else found --> generic metal
            # return "GenericMetal"
            return "NumTypes"

        else:
            # assume it's smina name
            for type_info in self.atom_type_data:
                if string == type_info.smina_name:
                    return type_info.sm
            # if nothing else found, return numtypes
            # technically not necessary to call this numtypes,
            # but including this here to make it equivalent to the cpp code
            return "NumTypes"

    def get_coords_and_types_info(self, mol):
        xs, ys, zs, atomic_nums, types = [], [], [], [], []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num == 1:
                continue
            else:
                smina_type = self.rdkitatom_to_smina_type(atom)
                if smina_type == "NumTypes":
                    type_int = self.n_features - 1
                else:
                    smina_type_int = self.atom_types.index(smina_type)
                    type_int = self.type_map[smina_type_int]
            x, y, z = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            xs.append(x)
            ys.append(y)
            zs.append(z)
            types.append(type_int)
            atomic_nums.append(atomic_num)

        return xs, ys, zs, types, atomic_nums

    def rdkitmol_to_df(self, input_file, mol_type="ligand"):
        if type(input_file) == str:
            input_file = Path(input_file)
        if input_file.suffix == ".parquet":
            return pd.read_parquet(input_file)
        mol = self.read_file(input_file)
        xs, ys, zs, types, atomic_nums = self.get_coords_and_types_info(mol)
        df = pd.DataFrame()
        df["x"], df["y"], df["z"] = xs, ys, zs
        df["atomic_number"] = atomic_nums
        df["types"] = types
        df["bp"] = int(mol_type == "receptor")
        return df
