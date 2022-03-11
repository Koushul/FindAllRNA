from Bio import SeqIO
import numpy as np
import pandas as pd
import RNA
import swifter
from encoders import tri_nucleotide_composition, shape_count, randomize, pad, TRINUCLEOTIDES

## Normalization of MFE with sq length https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0113380
ref_table = pd.read_csv('Dataset_S1.txt', sep='\t').to_dict()
EXPECTED_MFE = dict(zip(ref_table['L (nt)'], ref_table['MFE (kcal/mol)']))

def normalize_mfe(seq, mfe, scale=1.0):
    Lo = 8
    L = len(seq)
    return scale*(mfe - EXPECTED_MFE[L])/(L-Lo)

def encode(filename="test.fasta", random_only=False, seq=None):
    if seq is not None:
        seqs = [seq]
        origins = ['user']
    else:
        records = list(SeqIO.parse(filename, "fasta"))
        
        seqs = [str(r.seq) for r in records if "N" not in str(r.seq)]
        origins = [str(r.id) for r in records if "N" not in str(r.seq)]
    
    # seqs = pad(seqs)
    print(f'Processing {len(seqs)} sequences...\n')
    
    
    
    if random_only:
        seqs = randomize(seqs)
        # origins = ['random' for i in origins]
    
    df = pd.DataFrame(columns=['seqs', 'origins', 'SS', 'MFE'])
    
    df.seqs = seqs
    df.origins = origins
    df['length'] = df.seqs.apply(lambda x: len(x))
    
    print('[1/5] Computing Minimum Free Energy and Structure')
    rnafold = np.column_stack(df.seqs.swifter.apply(lambda x: RNA.fold(x)))
    
    df.MFE = rnafold[1] 
    df.MFE = df.MFE.astype(float)
    
    print('[2/5] Normalizing Minimum Free Energy')
    
    df['nMFE'] = df[['seqs', 'MFE']].apply(lambda x: normalize_mfe(x[0], x[1]), axis=1)
    df.SS = rnafold[0]
    
    df['GC'] = df.seqs.apply(lambda x: (x.count("G") + x.count("C")) / len(x)) 
    
    print('[3/5] Computing tri-nucleotide composition')
    comp = [list(tri_nucleotide_composition(seq).values()) for seq in seqs]
    
    print('[4/5] Extracting shapes from structures')
    
    shc = [shape_count(ss) for ss in df.SS]
    
    print('[5/5] Finalizing features')
    
    df = pd.concat([df.set_index('seqs'), 
            pd.DataFrame(comp, index = seqs, columns=TRINUCLEOTIDES), 
            pd.DataFrame(shc, index = seqs)], axis=1)
    
    df = df.reset_index()
    
    return df    





# from pprint import pprint
# # f = 'test.fasta'
# # f = '/tmp/nfs.fa'
# f = '/tmp/ncrna.fa'

# df = encode(f, random_only=True)
# print(df)
# df.to_csv('ncrna_shuffled.csv', index = False)
# print()

# # df = encode(f, random_only=True)

# # print(df)
# # # df.to_csv('rfam_shuffled.csv', index = False)
# # print()
    
    