from sklearn.preprocessing import OneHotEncoder
import numpy as np

aa_list = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale
hydro_scale = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}

def encode_hydrophobicity(seq, scale=hydro_scale):
    # Per-residue values
    values = [scale[aa] for aa in seq if aa in scale]
    return np.mean(values), np.std(values), np.max(values), np.min(values)

def aa_composition(seq):
    counts = [seq.count(aa) / len(seq) for aa in aa_list]
    return np.array(counts)

def one_hot_encode(seq, max_seq_len):
    mapping = {aa:i for i, aa in enumerate(aa_list)}
    arr = np.zeros((max_seq_len, len(aa_list)), dtype=int)
    for i, aa in enumerate(seq):
        arr[i, mapping[aa]] = 1
    return arr.flatten()

class SeqFeatures:
    def __init__(self, seqs):
        self.seqs = seqs
        self._max_seq_len = max(len(s) for s in seqs)

    @property
    def one_hot_encode(self):
        return np.array([one_hot_encode(s, self._max_seq_len) for s in self.seqs])
    
    @property
    def aa_composition(self):
        return np.array([aa_composition(s) for s in self.seqs])
    
    @property
    def hydro_features(self):
        return np.array([encode_hydrophobicity(seq) for seq in self.seqs])
    
    def get_feature_combination(self, features):
        feature_arrays = [getattr(self, feat) for feat in features]
        return np.hstack(feature_arrays)
    
