# Imports 
import numpy as np



# Padding function (to enable feeding all sentences together to the training function of neural networks)
def pad_sequence_data(encoded_data):

    max_len = max([len(sent) for sent in encoded_data])
        
    result = []
        
    for sent in encoded_data:
            
        for dif in range(max_len - len(sent)):
            sent = np.append(sent, np.array([0 for _ in range(sent.shape[1])]).reshape(1, sent.shape[1]), axis = 0)
                
        result.append(sent)
            
    return np.array(result)




# Suffle function for arrays
def unison_shuffled_copies(a, b):
    
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    
    return a[p], b[p]