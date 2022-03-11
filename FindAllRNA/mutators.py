import numpy as np
import RNA
# from Levenshtein import distance
from collections import defaultdict
from utils import normalize_mfe
from Bio import SeqIO
from tqdm import tqdm


class Mutator(object):
    
    def __init__(self, seqs):
        self.seqs = seqs
        self.mutants = defaultdict(str)
        for i in self.seqs:
            self.mutants[i] = []
        
    
    
    def mutate(self):        
        for s in self.seqs:
            for i in range(len(s)):
                self.mutants[s].append(s[:i] + s[i+1:])
    
    def compute_mfe(self):
        
        for seq, mut in tqdm(self.mutants.items()):
            ssi, _ = RNA.fold(str(seq))
            for mm in mut:
                ss, mfe = RNA.fold(mm)
                # yield ss, round(100.0*float(distance(ssi, ss))/len(seq), 1)
                # yield ss, mfe/len(seq)
                yield ss, normalize_mfe(seq, mfe)
                
    







if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # seqs = ["GCCGAAGTGGTGGAATTGGTAGACACGCTAGACTCAAAATCTGGTGGGAGCAATCCCGTGTCGGTTCGAGTCCGACCTTCGGCACCA", 
    #         "GCCTATATAGCTCAGAGGCAGAGCACTTCCTTGGTAAGGAAGAGGTCGGCGGTTCAATTCCGCTTATAGGCTCCA", 
    #         "CGGGAATAGCTCAGTTGGCTAGAGCATCAGCCTTCCAAGCTGAGGGTCGCGGGTTCGAGTCCCGTTTCCCGCTC"]
    
    records = list(SeqIO.parse('/tmp/ncrna.fa', "fasta"))
    seqs = [str(r.seq) for r in records if "N" not in str(r.seq)][:100]
    m = Mutator(seqs)
    m.mutate()
    mfes = np.array(list(m.compute_mfe()))
    
    sns.kdeplot([float(i[1]) for i in mfes], label='real')
    
    
    records = list(SeqIO.parse('/tmp/nfs.fa', "fasta"))
    seqs = [str(r.seq) for r in records if "N" not in str(r.seq)][:100]
    m = Mutator(seqs)
    m.mutate()
    mfes = np.array(list(m.compute_mfe()))
    
        
    sns.kdeplot([float(i[1]) for i in mfes], label='random')
    
    plt.legend()
    plt.show()