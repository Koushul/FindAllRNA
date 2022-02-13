from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def get_labels(fasta_file: str) -> list:
    seq_rec = list(SeqIO.parse(fasta_file, "fasta"))
    labels = [r.name for r in seq_rec]
    return labels

def write_seqs(fasta_file: str, seqs: list, ids:list, desc: str = ''):
    seqrecords = []
    for i, s in enumerate(seqs):
        sr = SeqRecord(s, id=ids[i], description=desc)
        seqrecords.append(sr)
    SeqIO.write(seqrecords, fasta_file, 'fasta')
    
class DataLoader(object):
    
    def __init__(self, x_train: str, x_test: str, x_val: str):
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val

        self.train_labels = get_labels(self.x_train)
        self.val_labels = get_labels(self.x_val)
        self.test_labels = get_labels(self.x_test)
        
        
        
        
        
    
    # def
    #     bn = 25
    #     seqTrain = get_seqs_with_bnoise(fastaTrain,nperc=bn)
    #     seqVal = get_seqs_with_bnoise(fastaVal,nperc=bn)
    #     seqTest = get_seqs_with_bnoise(fastaTest,nperc=bn)
            
            
        
        
        