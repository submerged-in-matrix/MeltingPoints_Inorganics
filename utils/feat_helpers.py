from env.modules import *

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
def count_bonds(mol):
    num_double = 0
    num_triple = 0
    num_aromatic = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            num_double += 1
        elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
            num_triple += 1
        elif bond.GetIsAromatic():
            num_aromatic += 1
    return num_double, num_triple, num_aromatic

def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan]*9 + [0]*128
    num_double, num_triple, num_aromatic = count_bonds(mol)
    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        num_double,
        num_triple,
        num_aromatic,
        Descriptors.FractionCSP3(mol)
    ]
    # Morgan fingerprint using MorganGenerator
    fp = morgan_gen.GetFingerprint(mol)
    features += list(fp)
    return features