from sklearn.preprocessing import OneHotEncoder
import numpy as np

aa_list = list("ACDEFGHIKLMNPQRSTVWY")

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
    
