from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
from sklearn.utils import shuffle

from encoders import NoisyEncoder


# Rfam families and their classes

snRNA = ['RF00003', 'RF00004', 'RF00007', 'RF00012', 'RF00015', 'RF00016', 'RF00020', 'RF00026', 'RF00066',
'RF00097', 'RF00149', 'RF00156', 'RF00191', 'RF00309', 'RF00321', 'RF00409', 'RF00432', 'RF00548',
'RF00560', 'RF00561', 'RF00619', 'RF01210']

cis_reg = ['RF00037', 'RF00050', 'RF00059', 'RF00080', 'RF00162', 'RF00167', 'RF00168', 'RF00174', 'RF00234',
'RF00379', 'RF00380', 'RF00391', 'RF00442', 'RF00485', 'RF00504', 'RF00515', 'RF00521', 'RF00524',
'RF00557', 'RF01051', 'RF01055', 'RF01057', 'RF01068', 'RF01073', 'RF01497', 'RF01726', 'RF01731',
'RF01734', 'RF01750', 'RF02271', 'RF02913', 'RF02914']

miRNA = ['RF00104', 'RF00451', 'RF00639', 'RF00641', 'RF00643', 'RF00645', 'RF00865', 'RF00875', 'RF00876',
'RF00882', 'RF00886', 'RF00906', 'RF01059', 'RF01911', 'RF01942', 'RF02000', 'RF02096']

sRNA = ['RF00519', 'RF01687', 'RF01690', 'RF01699', 'RF01705', 'RF02924', 'RF03064']

Intron = ['RF00029', 'RF01998', 'RF01999', 'RF02001', 'RF02003', 'RF02012']

rRNA = ['RF00001', 'RF00002']

tRNA = ['RF00005', 'RF01852']

lookup = {}
for i in snRNA:
    lookup[i] = 'snRNA'
for i in cis_reg:
    lookup[i] = 'cis_reg'
for i in miRNA:
    lookup[i] = 'miRNA'
for i in sRNA:
    lookup[i] = 'sRNA'
for i in Intron:
    lookup[i] = 'intron'
for i in rRNA:
    lookup[i] = 'rRNA'
for i in tRNA:
    lookup[i] = 'tRNA'


def get_labels(fasta_file: str) -> list:
    seq_rec = list(SeqIO.parse(fasta_file, "fasta"))
    labels = [r.name for r in seq_rec]
    return np.array(labels)

def write_seqs(fasta_file: str, seqs: list, ids:list, desc: str = ''):
    seqrecords = []
    for i, s in enumerate(seqs):
        sr = SeqRecord(s, id=ids[i], description=desc)
        seqrecords.append(sr)
    SeqIO.write(seqrecords, fasta_file, 'fasta')
    
class DataLoader(object):
    
    def __init__(self, x_train: str, x_test: str, x_val: str, default_noise: float = 0.0):
        self.default_noise = default_noise
        
        self.noise_encoder_train = NoisyEncoder(fasta_file=x_train)
        self.noise_encoder_test = NoisyEncoder(fasta_file=x_test)
        self.noise_encoder_val = NoisyEncoder(fasta_file=x_val)
        
        self.train_labels = get_labels(x_train)
        self.val_labels = get_labels(x_val)
        self.test_labels = get_labels(x_test)
        
    def generate_training_data(self, noise: float = None):
        data, labels = shuffle(self.noise_encoder_train.encode(noise or self.default_noise), self.train_labels)  
        return data, labels      
    
    def generate_testing_data(self, noise: float = None):
        return self.noise_encoder_test.encode(noise or self.default_noise), self.test_labels        
       
    def generate_validation_data(self, noise: float = None):
        return self.noise_encoder_val.encode(noise or self.default_noise), self.val_labels        
  
  
  
if __name__ == '__main__':
    
    # TODO: use fixtures instead
    generator = DataLoader(
        '../datasets/x_train.fasta', 
        '../datasets/x_val.fasta', 
        '../datasets/x_test.fasta'
    )
    
    train_data, train_labels = generator.generate_training_data()
    
    print(train_data[:10])
    print(train_labels[:10])
        
        
        