import torch
import torch.nn.functional as F
from utils.py import sigmoid
import sentencepiece as spm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
threshold = 0.5

sp = spm.SentencePieceProcessor()
sp.load('tokenizer\m.model')

Pad_token = sp.pad_id()


class LSTM():
    def __init__(self,embd_dim, hidden_dims, num_classes,vocab_size, batch_size):
        #storing hyperparams within class as instance attributes
        self.hidden_dim = hidden_dims
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embd_dim = embd_dim
        self.batch_size = batch_size

        #initializing the weights and biases of the LSTM
        Wxh = torch.randn(self.embd_dim, self.hidden_dim * 4).to(device) / (self.embd_dim**0.5)
        Whh = torch.randn(self.hidden_dim, self.hidden_dim * 4).to(device) / (self.hidden_dim**0.5)
        W_out = torch.randn(self.hidden_dim, self.num_classes).to(device) / (self.hidden_dim**0.5)
        W_embed = torch.randn(self.vocab_size, self.embd_dim).to(device) * 0.01
        
        b = torch.zeros(self.hidden_dim*4,).to(device)
        b[self.hidden_dim:2*self.hidden_dim] = 1.0
        b_out = torch.zeros(self.num_classes).to(device)
        
        #storing weights and biases values in model.param dict
        self.params = {
            "W_embed" : W_embed,
            "Wxh" : Wxh,
            "Whh" : Whh,
            "W_out": W_out,

            "b" : b,
            "b_out": b_out 
        }
    
    # def linear(self ,x, w, b):
    #     out = x @ w + b
    #     cache = x, w, b
    #     return out, cache
    
    # def linear_backward(dout, cache):
    #     x,w,b = cache

    #     dx = dout @ w.T
    #     dw = x.T @ dout
    #     db = dout.sum(axis=0)

    #     return dx, dw,db

    
    
    def forward_step(self, x, prev_h, prev_c,Wxh,Whh,b):
        # an single forward step of an LSTM

        # retrieving weights and bias value from self/model.params
              
        
        # matrix multiplication between input and previous hidden state with respective weights + bias
        # using torch.hsplit to split the output dimension of matrix mul to 4 gates, with each gate dim placed horizontally in variable a
        a = torch.hsplit((x @ Wxh) + (prev_h @ Whh) + b, 4)
        
        # retrieving each column in variable a and assinging each to their respective gates within the LSTM
        ai = a[0]
        af = a[1]
        ao = a[2]
        ag = a[3]

        # applying activation fucntions that correspond to each gate
        i  = sigmoid(ai)
        f = sigmoid(af)
        o = sigmoid(ao)
        g = torch.tanh(ag)
        
        # working out the next cell and hidden state
        next_c = (prev_c * f ) + (g * i)
        next_h = torch.tanh(next_c) * o
        
        #caching these varibales for backprop
        cache = x, Wxh, Whh, prev_c, prev_h, i, f, g,o, next_c
        
        return next_h, next_c, cache

    def backward_step(self,dnext_h, dnext_c, cache):
        # retrieving cached varibales from the forward step
        x, Wx, Wh, prev_c, prev_h, i, f, g,o,next_c = cache
    
        # calculating the gradients for both current and previous cell state
        dnext_c += dnext_h * o * (1-(torch.tanh(next_c)**2))
        dprev_c = dnext_c * f

        # calculating the gradients for each LSTM Gate
        da0 = dnext_c * g * i * (1 - i)
        da1 = dnext_c * prev_c * f * (1-f)
        da2 = dnext_h * torch.tanh(next_c) * o * (1 - o)
        da3 = dnext_c * i * (1-g**2)
        da = torch.hstack((da0,da1,da2,da3))

        # calculating gradients for weights, bias and the previous hidden state         
        dx = da @ Wx.T
        dWx = x.T @ da
        dWh = prev_h.T @ da
        dprev_h = da @ Wh.T
        db = da.sum(axis = 0)
        
        return dx, dprev_h, dprev_c, dWx, dWh, db
    
    def lstm_forward(self,x, h0, Wx, Wh, b):
        N,T,D = x.shape

        # initializing the cell, hidden states and cache for backprop
        c, hs, cache = torch.zeros_like(h0), [h0], []
        # going through each timestep and excuting forward step
        for t in range(T):
            next_h, c, cache_t = self.forward_step(x[:,t,:], hs[-1], c, Wx,Wh,b)
            hs.append(next_h)
            cache.append(cache_t)
        # stacking together all the hidden states collected through each timestep, column wise
        h = torch.stack(hs[1:], axis =1)
        return h, cache
    
    def lstm_backward(self,dh,cache):
        (N,T,H), (_,D) = dh.shape, cache[0][0].shape
        _, H4 = cache[0][1].shape
        
        # initializing the shapes of each derivative 
        dx = torch.empty((N, T, D)).to(device)
        dh0 = torch.zeros((N, H)).to(device)
        dc0 = torch.zeros((N, H)).to(device)
        dWx = torch.zeros((D, H4)).to(device)
        dWh = torch.zeros((H, H4)).to(device)
        db = torch.zeros(H4).to(device)
        
        # stepping backwards, from end-to-start, backpropping through each timestep
        for t in range(T-1, -1, -1):
            # executing the backward step and + dh0 to the current step's hidden state grad dh
            dx_t, dh0, dc0, dWx_t, dWh_t, db_t = self.backward_step(dh0 + dh[:, t,:], dc0, cache[t])
            
            dx[:, t,:] = dx_t
            dWx += dWx_t
            dWh += dWh_t
            db += db_t

          
        return dx, dh0, dWx, dWh, db
    
    def temporal_affine_forward(self,x, w, b, mask):
        # computing scores on the last hidden state
        N,T,D= x.shape
        # two lines of code below is trying to find the last valid token before the seqeunce is padded
        # becuase we want to compute score on the last time step, most seq are padded therefore we need the last valid token
        l_val_ix = torch.clamp(mask.sum(dim=1) - 1, min=0) 
        # finding the last valid token for each example in batch, using torch.arange to help
        x_last = x[torch.arange(N),l_val_ix]
        
        #computing the unormalized logits 
        out = x_last @ w + b
        # converting the unormalized logits in to probabilities for binary classifiction using sigmoid function
        out = sigmoid(out)

        #caching variables for backprop
        cache = x, w, b, out, mask
        return out, cache
    
    def temporal_affine_backward(self,dout,cache):
        x, w, b, out, mask = cache
        N, T, D = x.shape
        M = b.shape[0]
        # derivative of the sigmoid function
        dsigmoid = dout * (out * (1-out))
        
        # using the last valid token to avoid padding tokens
        l_val_ix = torch.clamp(mask.sum(dim=1) - 1, min=0) 
        x_last = x[torch.arange(N),l_val_ix]
        
        # computing derivatives for the weights and biases
        dw = x_last.T @ dsigmoid
        db = dsigmoid.sum(0)
        dx_last = dsigmoid @ w.T
        dx = torch.zeros_like(x)
        # only count the gradients for the last token for each example
        dx[torch.arange(N),l_val_ix] = dx_last

        return dx, dw, db

    def temporal_binary_cross_entropy(self, y_pred, y_true, mask):
        # computing the loss which will be binary cross entropy 
        N,T = y_pred.shape
        
        epsilon = 1e-15

        # avoiding log of 0 
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        y_true = y_true.view_as(y_pred)
        #computing the loss
        loss = -torch.sum(y_true * torch.log(y_pred + epsilon) + (1-y_true)* torch.log(1-y_pred + epsilon))/N
        # working out the gradient from loss and divide by the number of examples
        dx = (y_pred - y_true) / N
        return loss, dx

    def word_embedding(self,x, W):
        #embedding input x via the weight embedding W using python's indexing 
        out = W[x]
        
        cache = x,W
        return out, cache
    def word_embedding_backward(self,dout, cache):
        x, W = cache
        dW = torch.zeros_like(W)
        for k in range(x.shape[0]):
            for j in range(x.shape[1]):
                ix = x[k,j]
                dW[ix] += dout[k,j]
        return dW

    def create_config(self,learning_rate,w):
        
        
        config = {
            'learning_rate': learning_rate,
            'beta1': 0.9,
            'beta2':0.999,
            'epsilon':1e-8,
            'm': torch.zeros_like(w),
            'v':torch.zeros_like(w),
            't' : 0
        }
        
        return config
        
    
    def loss(self, x, targets = None):
        mask = x != Pad_token
        grads = {}
        Wxh, Whh, b = self.params["Wxh"], self.params["Whh"], self.params["b"]
        W_out, b_out = self.params["W_out"], self.params["b_out"]
        W_embed = self.params["W_embed"]
        
        h0 = torch.zeros((self.batch_size, self.hidden_dim)).to(device)
        x, cache_emb = self.word_embedding(x,W_embed)
        h , cache_lstm = self.lstm_forward(x, h0, Wxh, Whh, b)
        scores, cache_aff = self.temporal_affine_forward(h,W_out,b_out,mask)
        if targets is None:
            loss = scores
            grads = None
        else:
            loss, dout = self.temporal_binary_cross_entropy(scores, targets,mask)
            dout, dW_out, db_out = self.temporal_affine_backward(dout, cache_aff)
            dout, dh0, dWx, dWh, db = self.lstm_backward(dout,cache_lstm) 
            dW_embed = self.word_embedding_backward(dout, cache_emb)
            
            
            grads ={
                "W_embed" : dW_embed,
                "Wxh" : dWx,
                "Whh" : dWh,
                "W_out" : dW_out, 
    
                "b" : db,
                "b_out" : db_out
            }


        return loss, grads

    def update_params(self,w,dw, config = None):
        
        
        next_w = None    
        beta1, beta2, eps, learning_rate = config['beta1'], config['beta2'], config['epsilon'], config['learning_rate']
        t,m,v = config['t'] + 1, config['m'], config['v']
        m = beta1 * m + (1-beta1) * dw
        v = beta2 * v + (1-beta2) * (dw*dw)
        m_hat = m/(1-beta1 **t)
        v_hat = v/(1-beta2 **t)
        w-= learning_rate * m_hat / (torch.sqrt(v_hat) + eps)
        config["t"] = t
        config["m"] = m
        config["v"] = v
       
        next_w = w 

        return next_w, config

    def reset_grads(self,grads):
        for g in grads:
            grads[g].zero_() 
        return grads

    def sample(self,x,bos_token, eos_token, pad_sequence, block_size):
        x = [bos_token] + sp.encode_as_ids(x) + [eos_token]
        tok_x = torch.tensor(pad_sequence(x, block_size, Pad_token)).unsqueeze(0)
        mask = tok_x != Pad_token 
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wxh'], self.params['Whh'], self.params['b']
        W_out, b_out = self.params['W_out'], self.params['b_out']

        h0 = torch.zeros((self.batch_size, self.hidden_dim)).to(device)
        c, hs, cache = torch.zeros_like(h0), [h0], []
        embed_x, cache_we = self.word_embedding(x,W_embed)
        for t in range(x.shape[1]):
            next_h, c , cache_rnn = self.forward_step(embed_x[:,t,:], hs[-1],c)
            hs.append(next_h)
            cache.append(cache_rnn)
        h = torch.stack(hs[1:], axis =1)
        label, out_cache = self.temporal_affine_forward(h,W_out, b_out,mask)
        

        results = []
        for i in range(label.shape[0]):
            out = label[i]
            if out >= threshold:
                pred = "Phishing"
            else: 
                pred = "Safe"

        return pred