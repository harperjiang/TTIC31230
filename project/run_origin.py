import utils
import numpy as np
import edf
from time import time
import pickle
import os

train_data, trcnt = utils.load_data_onechar('data/ptb.train.txt', False)
valid_data, vacnt = utils.load_data_onechar('data/ptb.valid.txt', False)
test_data, tecnt = utils.load_data_onechar('data/ptb.test.txt', False)

hidden_dim = 200
n_vocab = utils.n_vocab
batch = 50
parameters = []
model = 'model_LSTM.pkl'
eta = 0.5
decay = 0.9

inp = edf.Value()
np.random.seed(0)

edf.params = []
C2V = edf.Param(edf.xavier((n_vocab, hidden_dim)))

# forget gate
Wf = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
bf = edf.Param(np.zeros((hidden_dim)))
# input gate
Wi = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
bi = edf.Param(np.zeros((hidden_dim)))
# carry cell
Wc = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
bc = edf.Param(np.zeros((hidden_dim)))
# output cell
Wo = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
bo = edf.Param(np.zeros((hidden_dim)))
    
V = edf.Param(edf.xavier((hidden_dim, n_vocab)))

parameters.extend([C2V, Wf, bf, Wi, bi, Wc, bc, Wo, bo, V])


# load the trained model if exist
if os.path.exists(model):
    with open(model, 'rb') as f:
        p_value = pickle.load(f)
        idx = 0
        for p in p_value:
            parameters[idx].value = p
            idx += 1
                    

def LSTMCell(xt, h, c):
    
    f = edf.Sigmoid(edf.Add(edf.VDot(edf.ConCat(xt, h), Wf), bf))
    i = edf.Sigmoid(edf.Add(edf.VDot(edf.ConCat(xt, h), Wi), bi))
    o = edf.Sigmoid(edf.Add(edf.VDot(edf.ConCat(xt, h), Wo), bo))
    c_hat = edf.Tanh(edf.Add(edf.VDot(edf.ConCat(xt, h), Wc), bc))
    c_next = edf.Add(edf.Mul(f, c), edf.Mul(i, c_hat))
    h_next = edf.Mul(o, edf.Tanh(c_next))
            
    return h_next, c_next


def BuildModel():
 
    edf.components = []

    B = inp.value.shape[0]
    T = inp.value.shape[1]
    h = edf.Value(np.zeros((B, hidden_dim))) 
    c = edf.Value(np.zeros((B, hidden_dim)))
    
    score = []
    
    for t in range(T-1):
 
        wordvec = edf.Embed(edf.Value(inp.value[:,t]), C2V) 
        xt = edf.Reshape(wordvec, [-1, hidden_dim])
        h_next, c_next = LSTMCell(xt, h, c)
        p = edf.SoftMax(edf.VDot(h_next, V))
        logloss = edf.Reshape(edf.LogLoss(edf.Aref(p, edf.Value(inp.value[:,t+1]))), (B, 1))
        
        if t == 0:
            loss = logloss
        else:
            loss = edf.ConCat(loss, logloss)
            
        score.append(p)    
        h = h_next
        c = c_next
    
    masks = np.zeros((B, T-1), dtype = np.int32)
    masks[inp.value[:,1:] != 0] = 1
    #loss = edf.MeanwithMask(loss, edf.Value(masks)) 
    loss = edf.Mean(loss)
    
    return loss, score
    
    
def CalPerp(score):
    
    prob = [p.value for p in score]
    prob = np.transpose(np.stack(prob, axis = 0),(1,0,2))
    
    B = prob.shape[0]
    T = prob.shape[1]
    V = prob.shape[2]
    
    masks = np.zeros((B, T), dtype=np.int32)
    masks[inp.value[:,1:] != 0] = 1
    
    prob = prob.reshape(-1)
    idx = np.int32(inp.value[:,1:].reshape(-1))
    outer_dim = len(idx)
    inner_dim = len(prob)/outer_dim
    pick = np.int32(np.array(range(outer_dim))*inner_dim + idx)
    prob = prob[pick].reshape(B, T)
        
    return -np.sum(np.log(prob[np.nonzero(prob*masks)]))

def Predict(max_step, prefix):
   
    edf.components = []

    T = max_step       
    h = edf.Value(np.zeros((1, hidden_dim))) 
    c = edf.Value(np.zeros((1, hidden_dim))) 
    
    prediction = []

    for t in range(T):
   
        if t < len(prefix):
            pred = edf.Value(prefix[t])
            prediction.append(pred)              
        else:
            prediction.append(pred)

        wordvec = edf.Embed(pred, C2V)
        xt = edf.Reshape(wordvec, [-1, hidden_dim])
        h_next,c_next = LSTMCell(xt, h, c)
        p = edf.SoftMax(edf.VDot(h_next, V))
        pred = edf.ArgMax(p)
        h = h_next
        c = c_next   
            
    edf.Forward()
    
    idx = [pred.value for pred in prediction]
    stop_idx = utils.to_index('}')
    
    if stop_idx in idx:
        return idx[0:idx.index(stop_idx)+1]
    else:
        return idx

def Eval(data, cnt):
    
    perp = 0.
    avg_loss = 0.
    test_batches = range(0, len(data), batch)
    test_minbatches = [data[idx:idx+batch] for idx in test_batches]
    
    for minbatch in test_minbatches:
        
        x_padded = utils.make_mask(minbatch)
        inp.set(x_padded)
        loss, score = BuildModel()
        edf.Forward()
        avg_loss += loss.value
        perp += CalPerp(score)
           
    perp = np.exp(perp/cnt)
    avg_loss /= len(test_batches)
    return perp, avg_loss


############################################### training loop #####################################################

batches = range(0, len(train_data), batch)
minbatches = [train_data[idx:idx+batch] for idx in batches]

epoch = 30

# initial Perplexity and loss
perp, loss = Eval(valid_data, vacnt)
print("Initial: Perplexity: %0.5f Avg loss = %0.5f" % (perp, loss))    
best_loss = loss
prefix = 'the agreements bring'  
generation = Predict(400, utils.to_idxs(prefix))
print("Initial generated sentence ")
print (utils.to_string(generation))
    
    
for ep in range(epoch):

    perm = np.random.permutation(len(minbatches)).tolist() 
    stime=time()
    
    for k in range(len(minbatches)):
        
        minbatch = minbatches[perm[k]]
        x_padded = utils.make_mask(minbatch)
        inp.set(x_padded)
        loss, score = BuildModel()
        edf.Forward()
        edf.Backward(loss)
        edf.GradClip(10)
        edf.SGD(eta)
       
    duration = (time() - stime)/60.
    
    perp, loss = Eval(valid_data, vacnt)
    print("Epoch %d: Perplexity: %0.5f Avg loss = %0.5f [%.3f mins]" % (ep, perp, loss, duration))
    
    # generate some text given the prefix and trained model
    prefix = 'the agreements bring'  
    generation = Predict(400, utils.to_idxs(prefix))
    print("Epoch %d: generated sentence " % ep)
    print (utils.to_string(generation)) 

    if loss < best_loss:
        
        best_loss = loss
        # save the model
        f = open(model, 'wb')
        p_value = []
        for p in parameters:
            p_value.append(p.value)
        pickle.dump(p_value, f)
        
    else:
        
        # load the last best model and decay the learning rate
        eta *= decay
        with open(model, 'rb') as f:
            p_value = pickle.load(f)
            idx = 0
            for p in p_value:
                parameters[idx].value = p
                idx += 1
