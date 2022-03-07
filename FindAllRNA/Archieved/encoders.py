
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import random
import itertools
from textwrap import wrap
from collections import defaultdict

def TriNcleotideComposition(sequence, base="ATGC"):
    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    tnc_dict = {}
    for triN in trincleotides:
        tnc_dict[triN] = 0
    for i in range(len(sequence) - 2):
        tnc_dict[sequence[i:i + 3]] += 1
    for key in tnc_dict:
       tnc_dict[key] /= (len(sequence) - 2)
    return tnc_dict

class AbstractSequenceEncoder:
    ALLOWED_CHARACTERS = 'ATGC'
    
    def encode(self, seq: Seq):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError   
    
class AbstractFastaEncoder(AbstractSequenceEncoder):
    pass

class DummySeqEncoder(AbstractSequenceEncoder):
    
    def encode(self, seq: Seq):
        return seq
    
    
class DummyFastaEncoder(AbstractSequenceEncoder):
    
    def __init__(self, fasta_file: str):
        self.fasta_file = fasta_file
    
    def encode(self):
        seq_rec = list(SeqIO.parse(self.fasta_file, "fasta"))
        return seq_rec
    
class NoisyEncoder(AbstractFastaEncoder):
    
    def __init__(self, fasta_file: str):
        self.fasta_file = fasta_file
    
    def encode(self, nperc: float):
        """
        Pads sequences with random di-nucleotide preserving random segments at the head and tail.
        Returns a list of padded sequences.
        """
        seq_rec = list(SeqIO.parse(self.fasta_file, "fasta"))
        samples = []
        for i, r in enumerate(seq_rec):
            head = ''
            tail = ''
            if nperc > 0:
                sw = wrap(str(r.seq), 2) + wrap(str(r.seq)[1:], 2)
                head = np.random.choice(sw, int(0.25*len(r.seq)*nperc/100))
                tail = np.random.choice(sw, int(0.25*len(r.seq)*nperc/100))            
                head = ''.join(head)
                tail = ''.join(tail)
            samples.append(head+str(r.seq)+tail)
        return np.array(samples)
    
    
class RandomEncoder(AbstractFastaEncoder):
    
    def __init__(self, fasta_file: str):
        self.fasta_file = fasta_file
        self.archieve = defaultdict(set) #used for getting the original sequence and ID back
        
    def encode(self, k: int, x: int) -> list:
        """
        Returns a list of randomly shuffled sequences preserving the `k`-nucleotide frequency
        Suggested values for k are 2, 3, 6, 12
        Each sequence in the fasta file is shuffled `x` times
        """
        
        seq_rec = list(SeqIO.parse(self.fasta_file, "fasta"))
        samples=[]
        
        for _ in range(x):
            for i, r in enumerate(seq_rec):
                rseq = str(r.seq)
                sw = [rseq[i:i+k] for i in range(len(rseq)-k+1)]
                np.random.shuffle(sw)
                ss = Seq(''.join(sw))
                self.archieve[r.seq].add((ss, r.name))
                samples.append(str(ss))
        return samples
        
    def decode(self, seq):
        return self.archieve[seq]
    
    

class KMerEncoder(AbstractSequenceEncoder):
    
    def __init__(self, k: int, n: int, padding=None):
        self.k = k
        self.n = n
        assert padding in ['random', 'constant', None]
        self.padding = padding
        self.kmers = [''.join(x) for x in itertools.product(self.ALLOWED_CHARACTERS, repeat=self.k)] #all possible kmers
        self.char_to_int = dict((c, i) for i, c in enumerate(self.kmers)) #encodings
        
    
    def encode(self, seq):
        """
        Returns an integer encoding of the sequence of shape (k, n)
        Window skips instead of sliding
        """
    
        seq = seq[:self.n]
        
        if self.padding == 'random':
            seqr = np.random.randint(len(self.ALLOWED_CHARACTERS), size=self.n)
        elif self.padding == 'constant':
            seqr = np.zeros(self.n)
            
        data = wrap(str(seq), self.k)
        data = [d for d in data if len(d) == self.k]
        integer_encoded = [self.char_to_int[c] for c in data]   
        
        if self.padding == None:
            return integer_encoded
        else: 
            seqr[0:len(integer_encoded)] = integer_encoded
        
        return seqr
        
    
class OneHotEncoder(AbstractSequenceEncoder):
    
    def __init__(self, k: int, n: int, padding: str):
        self.k = k
        self.n = n
        assert padding in ['random', 'constant']
        self.padding = padding
        self.kmers = [''.join(x) for x in itertools.product(self.ALLOWED_CHARACTERS, repeat=self.k)] #all possible kmers
        self.char_to_int = dict((c, i) for i, c in enumerate(self.kmers)) #encodings
    
    def encode(self, seq: Seq):
        
        seq = seq[:self.n]
        
        if self.padding == 'random':
            seqr = np.random.randint(len(self.ALLOWED_CHARACTERS), size=self.n)
        elif self.padding == 'constant':
            seqr = np.zeros(self.n)
            
        data = [str(seq)[i:i+self.k] for i in range(len(seq)-self.k+1)]
        integer_encoded = [self.char_to_int[c] for c in data]
        seqr[0:len(integer_encoded)] = integer_encoded
        
        return seqr

                



if __name__ == '__main__':
    # s = Seq('TTATGACCC')
    # encoder = KMerEncoder(2, 50, 'constant')
    # e = encoder.encode(s)
    # print(s)
    # print(encoder.char_to_int)
    # print(e)
    
    # encoder = KMerEncoder(3, 50, 'random')
    # e = encoder.encode(s)
    # print(s)
    # print(encoder.char_to_int)
    # print(e)
    
    # encoder = OneHotEncoder(2, 50, 'constant')
    # e = encoder.encode(s)
    # print(s)
    # print(encoder.char_to_int)
    # print(e)
    
    # from pprint import pprint
    # e = RandomEncoder('../datasets/fixture.fasta')
    # s = e.encode(2, 2)
    # print(len(s))
    # print(s)
    # print()
    # pprint(e.archieve)
    
    
    from pprint import pprint
    e = NoisyEncoder('../datasets/fixture.fasta')
    s = e.encode(10)

    pprint(s)
    