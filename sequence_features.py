from sklearn.preprocessing import OneHotEncoder
import numpy as np

aa_list = list("ACDEFGHIKLMNPQRSTVWY")

def one_hot_encode(seq):
    mapping = {aa:i for i, aa in enumerate(aa_list)}
    arr = np.zeros((len(seq), len(aa_list)), dtype=int)
    for i, aa in enumerate(seq):
        arr[i, mapping[aa]] = 1
    return arr.flatten()

def aa_composition(seq):
    counts = [seq.count(aa) / len(seq) for aa in aa_list]
    return np.array(counts)

class SeqFeatures:
    def __init__(self, seqs):
        self.seqs = seqs

    @property
    def one_hot_encode(self):
        return np.array([one_hot_encode(s) for s in self.seqs])
    
    @property
    def aa_composition(self):
        return np.array([aa_composition(s) for s in self.seqs])