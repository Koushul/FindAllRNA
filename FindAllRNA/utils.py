from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
import requests
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from tqdm import tqdm
import json

from encoders import NoisyEncoder, DummySeqEncoder, DummyFastaEncoder, AbstractSequenceEncoder, AbstractFastaEncoder


# Rfam families and their classes
with open('rfam.json', 'r') as j:
     rfam = json.loads(j.read())


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
        
        # self.seq_encoder = DummySeqEncoder()
        # self.fasta_encoder = DummyFastaEncoder()
        
    def attach_sequence_encoder(self, encoder: AbstractSequenceEncoder):
        assert isinstance(encoder, AbstractSequenceEncoder)
        self.seq_encoder = encoder
        
    
    def attach_fasta_encoder(self, encoder: AbstractFastaEncoder):
        assert isinstance(encoder, AbstractFastaEncoder)
        self.fasta_encoder = encoder
        
        
    def generate_training_data(self, noise: float = None):
        data, labels = shuffle(self.noise_encoder_train.encode(noise or self.default_noise), self.train_labels)  
        return data, labels      
    
    def generate_testing_data(self, noise: float = None):
        return self.noise_encoder_test.encode(noise or self.default_noise), self.test_labels        
       
    def generate_validation_data(self, noise: float = None):
        return self.noise_encoder_val.encode(noise or self.default_noise), self.val_labels        
  
  
class MonteCarloTumbler(object):
    """
    Given a trained model, estimates the uncertainties associated with its predictions.
    Implements `Dropout as a bayesian Approximation: Representing Model Uncertainty in Deep Learning (arxiv.org/abs/1506.02142)`
    """
    ## Code adapted from https://github.com/bioinformatics-sannio/ncrna-deep/
    
    
    def __init__(self, trained_model: Sequential):
        assert isinstance(trained_model, Sequential)
        self.trained_model = trained_model
        self.model_mc = K.function([self.trained_model.input, K.learning_phase()], [self.trained_model.output])
    
  
    def tumble(self, random_sequences: list, functional_sequences: list, iterations: int = 100):
        num_classes = int(self.trained_model.output.shape[1])
        
        avrp_rnd = np.zeros((len(random_sequences),num_classes))
        avrp_nornd = np.zeros((len(functional_sequences),num_classes))

        fp_rnd = np.zeros((len(random_sequences),num_classes))
        fp_nornd = np.zeros((len(functional_sequences),num_classes))

        avrhp_rnd = np.zeros((len(random_sequences)))
        avrhp_nornd = np.zeros((len(functional_sequences)))

        p_rnd = np.zeros((iterations,len(random_sequences),num_classes))
        p_nornd = np.zeros((iterations,len(functional_sequences),num_classes))

        for i in tqdm(range(iterations)):
            preds_nornd=self.model_mc([functional_sequences,1])
            p_nornd[i,:,:] = preds_nornd[0]
            avrp_nornd = avrp_nornd + preds_nornd[0]
            avrhp_nornd = avrhp_nornd + np.sum(-preds_nornd[0]*np.log2(preds_nornd[0]+1e-10),1)
            midx = np.argmax(preds_nornd[0],1)
            for j in range(len(functional_sequences)):
                fp_nornd[j,midx[j]] = fp_nornd[j,midx[j]] + 1

            preds_rnd=self.model_mc([random_sequences,1])
            p_rnd[i,:,:] = preds_rnd[0]
            avrp_rnd = avrp_rnd + preds_rnd[0]
            avrhp_rnd = avrhp_rnd + np.sum(-preds_rnd[0]*np.log2(preds_rnd[0]+1e-10),1)
            midx = np.argmax(preds_rnd[0],1)
            for j in range(len(random_sequences)):
                fp_rnd[j,midx[j]] = fp_rnd[j,midx[j]] + 1


        avrp_rnd /= iterations
        avrp_nornd /= iterations
        fp_rnd /= iterations
        fp_nornd /= iterations
        avrhp_rnd /= iterations
        avrhp_nornd /= iterations

        # compute indicators entropy (hp) variance (var) max prob (maxp) and f max

        self.hp_nornd = np.sum(-avrp_nornd*np.log2(avrp_nornd+1e-10),1)
        self.hp_rnd = np.sum(-avrp_rnd*np.log2(avrp_rnd+1e-10),1)

        # self.var_rnd = np.var(p_rnd,0)
        # self.var_nornd = np.var(p_nornd,0)

        # self.orderp_rnd = np.argsort(-avrp_rnd,1)
        # self.orderp_nornd = np.argsort(-avrp_nornd,1)

        # self.maxp_nornd = np.max(avrp_nornd,1)
        # self.maxp_rnd = np.max(avrp_rnd,1)
        # self.maxfp_nornd = np.max(fp_nornd,1)
        # self.maxfp_rnd = np.max(fp_rnd,1)
        
  
    def find_threshold(self):
        
        def acc(T):
            return np.where(self.hp_rnd > T)[0].__len__() / len(self.hp_rnd), np.where(self.hp_nornd <= T)[0].__len__() / len(self.hp_nornd)

        x = np.arange(0, 10, 0.01)
        ys = np.array([acc(i) for i in x])
        yr = ys[:,0]
        yn = ys[:,1]

        idx = [np.where(np.round(yr, 2) == np.round(yn, 2))[0]]
        threshold, accuracy = np.mean(x[idx]), np.mean(yr[idx])
        
        return threshold
  
  
  
  
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
        
        
        