import numpy as np
from Bio import PDB

from amino_acid_properties import *

def pdb_extractor(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    ppb = PDB.PPBuilder()
    sequence = []
    residues_properties = []
    ca_atoms = []

    for model in structure:
        for chain in model:
            peptides = ppb.build_peptides(chain)
            for pp in peptides:
                peptide_sequence = pp.get_sequence()
                sequence.append(str(peptide_sequence))

                for i, residue in enumerate(pp):
                    residue_name = residue.get_resname()
                    properties = {
                        'Residue': AMINOACIDMAP.get(residue_name),
                        'Relative Position': (i + 1) / len(pp),
                    }
                    residues_properties.append((i, properties))
                    if "CA" in residue:
                        ca_atoms.append(residue["CA"])

    full_sequence = "".join(sequence)

    n = len(ca_atoms)
    distance_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            coord1 = ca_atoms[i].get_coord()
            coord2 = ca_atoms[j].get_coord()
            distance = np.linalg.norm(coord1 - coord2)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    interaction_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            hyd1 = HYDROPHOBICITY.get(full_sequence[i], 0.0)
            hyd2 = HYDROPHOBICITY.get(full_sequence[j], 0.0)
            interaction_matrix[i][j] = interaction_matrix[j][i] = abs(hyd1 - hyd2)

    return {
        "Full Sequence": full_sequence,
        "Residue Properties": residues_properties,
        "Edge Properties": interaction_matrix,
        "Distance matrix": distance_matrix
    }
