import numpy as np
# from Levenshtein import distance
from collections import defaultdict
from Bio import SeqIO
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import namedtuple
import sys
import pandas as pd
from encoders import shape_count
from Bio import pairwise2

## Normalization of MFE with sq length https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0113380
ref_table = pd.read_csv('/Users/koushulramjattun/Downloads/FindAllRNA/FindAllRNA/Dataset_S1.txt', sep='\t').to_dict()
EXPECTED_MFE = dict(zip(ref_table['L (nt)'], ref_table['MFE (kcal/mol)']))

def normalize_mfe(seq, mfe, scale=1.0):
    Lo = 8
    L = len(seq)
    return scale*(mfe - EXPECTED_MFE[L])/(L-Lo)


"""
>seq_0_mutant_0 ncrna
CCGGAGUGGUGAAACUGGUAGACGCGCUAGACUCAAAAUCUAGUAAGGGCAACCUUGUGUCGGUUCGAGUCCGACCUUCGGCACCA
(((((..(((...(((.(..(((((((((((.......))))))((((....)))))))))....).)))...))))))))..... (-24.50)
"""

FoldedMutant = namedtuple('FoldedMutant', 'label seq structure mfe')

def read_fold(filename):
    from itertools import islice
    
    f = open(filename, 'r')
    while True:
        next_chunk = list(islice(f, 3))
        if not next_chunk:
            break
        else:
            label, seq, ss_mfe = [x.strip() for x in next_chunk]
            try:
                ss, mfe = ss_mfe.split(' ')
                mfe = mfe[1:-1]
            except Exception as e:
                ss, mfe = ss_mfe.split('( ')
                mfe = mfe[:-1]
            nmfe = normalize_mfe(seq, float(mfe[1:-1]), scale=100)
            # nmfe = float(mfe[1:-1]) / len(seq)
            yield FoldedMutant(label=label[1:].split(' ')[0], seq=seq, structure=ss, mfe=nmfe)
    
    f.close()
        
    

class Mutator(object):
    """
    Takes a list of DNA/RNA sequences and
    generates a list of mutants
    """
    
    def __init__(self, seqs, rna_type):
        self.seqs = seqs
        self.rna_type = rna_type
        self.mutants = defaultdict(str)
        for i in self.seqs:
            self.mutants[i] = {'deletions': [], 
                    'substitutions': []}
        self.master = []
        
    def mutate(self, deletions=True, substitutions=True):  
        pbar = tqdm(total=len(self.seqs))      
        for j, s in enumerate(self.seqs):
            for i in range(len(s)):
                head, tail, neck = s[:i], s[i+1:], s[i]
                if deletions:
                    mut_seq = head + tail
                    seq_rec = SeqRecord(Seq(mut_seq), id=f'seq_{j}_deletion_{i}', description=self.rna_type)
                    self.mutants[s]['deletions'].append(mut_seq)
                    self.master.append(seq_rec)
                if substitutions:
                    for x in ['A', 'T', 'G', 'C']:
                        if neck != x:
                            mut_seq = head + x + tail
                            seq_rec = SeqRecord(Seq(mut_seq), id=f'seq_{j}_substitution_{neck}|{x}', description=self.rna_type)
                            self.mutants[s]['substitutions'].append(mut_seq)
                            self.master.append(seq_rec)
            pbar.update()
            # pbar.set_description(f"{np.random.choice(['A', 'T', 'G', 'C'])} >> {np.random.choice(['A', 'T', 'G', 'C'])}") ## just for show
            
        pbar.close()
                            
                    
    
    def save_mutants(self, name, selection='all'):
        if selection == 'all': 
            targets = self.master       
            
        elif selection == 'deletions':
            targets = [t for t in self.master if 'deletion' in t.id]
            
        elif selection == 'substitutions':
            targets = [t for t in self.master if 'substitution' in t.id]
            
        SeqIO.write(targets, name, 'fasta')
        print(f'Wrote {len(targets)} sequences to {name}')
    



if __name__ == '__main__':
    from pprint import pprint
    
    records = list(SeqIO.parse('./sources/shuffled_rfam250.fa', "fasta"))
    print(f'Mutating 👾 {len(records)} sequences')
    m = Mutator([r.seq for r in records], rna_type='shuffled')
    m.mutate()
    m.save_mutants('./mutated/mutated_shuffled_rfam250.fa')
    
        