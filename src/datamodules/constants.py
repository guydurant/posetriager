# Taken from Smina
ADNAMES = {
    "Hydrogen": "H",
    "PolarHydrogen": "HD",
    "AliphaticCarbonXSNonHydrophobe": "C",
    "AliphaticCarbonXSHydrophobe": "C",
    "AromaticCarbonXSNonHydrophobe": "A",
    "AromaticCarbonXSHydrophobe": "A",
    "Nitrogen": "N",
    "NitrogenXSDonor": "N",
    "NitrogenXSAcceptor": "NA",
    "NitrogenXSDonorAcceptor": "NA",
    "NitrogenXSAcceptor": "NA",
    "Oxygen": "O",
    "OxygenXSDonor": "O",
    "OxygenXSDonorAcceptor": "OA",
    "OxygenXSAcceptor": "OA",
    "Sulfur": "S",
    "SulfurAcceptor": "SA",
    "Phosphorus": "P",
    "Fluorine": "F",
    "Chlorine": "Cl",
    "Bromine": "Br",
    "Iodine": "I",
    "Magnesium": "Mg",
    "Manganese": "Mn",
    "Zinc": "Zn",
    "Calcium": "Ca",
    "Iron": "Fe",
    "GenericMetal": "M",
}

# Taken from Smina
ATOM_TYPES = """#Name radius depth solvation volume covalent_radius xs_radius xs_hydrophobe xs_donor xs_acceptr ad_heteroatom
AliphaticCarbonXSHydrophobe 2 0.15 -0.00143 33.5103 0.77 1.9 1 0 0 0
AliphaticCarbonXSNonHydrophobe 2 0.15 -0.00143 33.5103 0.77 1.9 0 0 0 0
AromaticCarbonXSHydrophobe 2 0.15 -0.00052 33.5103 0.77 1.9 1 0 0 0
AromaticCarbonXSNonHydrophobe 2 0.15 -0.00052 33.5103 0.77 1.9 0 0 0 0
Nitrogen 1.75 0.16 -0.00162 22.4493 0.75 1.8 0 0 0 1
NitrogenXSDonor 1.75 0.16 -0.00162 22.4493 0.75 1.8 0 1 0 1
NitrogenXSDonorAcceptor 1.75 0.16 -0.00162 22.4493 0.75 1.8 0 1 1 1
NitrogenXSAcceptor 1.75 0.16 -0.00162 22.4493 0.75 1.8 0 0 1 1
Oxygen 1.6 0.2 -0.00251 17.1573 0.73 1.7 0 0 0 1
OxygenXSDonor 1.6 0.2 -0.00251 17.1573 0.73 1.7 0 1 0 1
OxygenXSDonorAcceptor 1.6 0.2 -0.00251 17.1573 0.73 1.7 0 1 1 1
OxygenXSAcceptor 1.6 0.2 -0.00251 17.1573 0.73 1.7 0 0 1 1
Sulfur 2 0.2 -0.00214 33.5103 1.02 2 0 0 0 1
SulfurAcceptor 2 0.2 -0.00214 33.5103 1.02 2 0 0 0 1
Phosphorus 2.1 0.2 -0.0011 38.7924 1.06 2.1 0 0 0 1"""
