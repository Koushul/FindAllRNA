from textwrap import wrap
import numpy as np
from shapes import MDG_Stem, parseBracketString
from collections import defaultdict
import sys
from ushuffle import shuffle, Shuffler
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord



BASE = "ATGC"
TRINUCLEOTIDES = [nn1 + nn2 + nn3 for nn1 in BASE for nn2 in BASE for nn3 in BASE]
SHAPES = ["Hairpin", "Inner", "Stems", "Multiloop"]

DEFOUT = open('/dev/null', 'w')

def tri_nucleotide_composition(sequence: str):
    assert isinstance(sequence, str)
    tnc_dict = {}
    for triN in TRINUCLEOTIDES:
        tnc_dict[triN] = 0
    for i in range(len(sequence) - 2):
        tnc_dict[sequence[i:i + 3]] += 1
    for key in tnc_dict:
       tnc_dict[key] /= (len(sequence) - 2)
    return tnc_dict


def shape_count(ss):
    structures = defaultdict(int)
    for s in SHAPES:
        structures[s] = 0
    
    contacts = list(parseBracketString(ss))
    
    if len(contacts) == 0:
        return structures
    
    x = MDG_Stem()

    x.assemble(contacts, outp=DEFOUT)
    for motif in x.find_all_motifs():
        structures[motif[0]]+=len(motif[1:])
        
    e = contacts[0][0]
    for a, b in contacts:
        if a != e:
            structures['Stems']+=1
            e = a
        e += 1 
    
    return structures

def pad(seqs):
    padded = []
    for seq in seqs:
        head = ''
        tail = ''
        size = np.random.choice([2, 3, 6, 12, 18])
        sw = wrap(seq, 2) + wrap(seq[1:], 2)
        head = np.random.choice(sw, size)
        tail = np.random.choice(sw, size)            
        head = ''.join(head)
        tail = ''.join(tail)
        padded.append(head+str(seq)+tail)
        
    return padded
        

def randomize(seqs, n=1):
    shuffled = []
    for seq in seqs:
        k = np.random.choice([2, 3, 6])
        shuffler = Shuffler(str.encode(seq), k)
        for i in range(n):
            seqres = shuffler.shuffle()
            s = shuffle(str.encode(seq), k).decode()
            shuffled.append(s)
    return shuffled
    



if __name__ == '__main__':
    from pprint import pprint
    import pandas as pd
    
    records = list(SeqIO.parse('./sources/rfam250.fa', "fasta"))
    rseqs = randomize([str(r.seq) for r in records])    
    SeqIO.write([SeqRecord(Seq(r), id=f'seq_{idx}', description='random') for idx, r in enumerate(rseqs)], './sources/shuffled_rfam250.fa', 'fasta')    
    
    
    

    