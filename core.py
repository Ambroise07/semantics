import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# Variables globales gérées dynamiquement
words = []
stoi = {}
itos = {}
vocab_size = 0

block_size, dime_ = 3, 2

#------------------------ the tokenizer ------------------------------#
def build_vocab(rawdata):
    global words, stoi, itos, vocab_size
    random.seed(12389)
    
    words = open(rawdata, 'r').read().splitlines()
    random.shuffle(words)

    chars = sorted(list(set(''.join(words))))
    stoi = {s: i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}  
    vocab_size = len(stoi)

    return words, stoi, itos

#---------------------input and output tensors------------------------#
def build_dataset(words_list, block_size:int=block_size):
    X, Y = [], []
    for w in words_list:
        target_x = [0] * block_size 
        for char in w + '.':
            int_ = stoi[char]
            X.append(target_x)
            Y.append(int_)
            target_x = target_x[1:] + [int_]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

#---------------------- building neuronal network ------------------------#
gen = torch.Generator().manual_seed(123490333340)

# Paramètres globaux par défaut (réinitialisés dynamiquement via init_network)
C = torch.randn(28, dime_, generator=gen)
W1 = torch.randn((block_size * dime_, 500), generator=gen)
B1 = torch.randn(500, generator=gen)
W2 = torch.randn((500, 28), generator=gen)
B2 = torch.randn(28, generator=gen)

parameters = [C, W1, W2, B1, B2]

for p in parameters:
    p.requires_grad = True


def init_network(v_size):
    """ Initialisation dynamique selon la méthode row * col """
    global C, W1, B1, W2, B2, parameters
    
    C = torch.randn(v_size, dime_, generator=gen)
    
    
    row = block_size
    col = dime_
    
    W1 = torch.randn((row * col, 500), generator=gen)
    B1 = torch.randn(500, generator=gen)
    
    W2 = torch.randn((500, v_size), generator=gen)
    B2 = torch.randn(v_size, generator=gen)
    
    parameters = [C, W1, W2, B1, B2]
    for p in parameters:
        p.requires_grad = True


# ------------------------ training part --------------------------------#
def train(XTr, YTr, iter=200000, ouput:str="lr"):
    if ouput.lower() not in ['lr', 'steps']:
        raise TypeError('ouput should be lr or steps.')
      
    lrs = None
    if ouput.lower() == 'lr': 
        lr_epochs = torch.linspace(-3, 0, iter)
        lrs = 10**lr_epochs
      
    axis = []
    loss_items = []

    for _ in range(iter):
        # Minibatch de taille 32
        idx = torch.randint(0, XTr.shape[0], (32,))
    
        embed = C[XTr[idx]]
        H = torch.tanh(embed.view(-1, block_size * dime_) @ W1 + B1)
        logits = H @ W2 + B2
        loss = F.cross_entropy(logits, YTr[idx]) # Correction de la perte (pas de signe négatif)

        for p in parameters:
            p.grad = None
        loss.backward()

        if lrs is None:
            lr = .0396
            for p in parameters:
                p.data += -lr * p.grad
            axis.append(_)
            loss_items.append(loss.item())
        else:
            lr = lrs[_].item()
            for p in parameters:
                p.data += -lr * p.grad
            axis.append(lr) # Placé en dehors de la boucle des paramètres
            loss_items.append(loss.item())

    return axis, loss_items  


#--------------------- display utilities ------------------------------#
def plot_fig(x, y):
    plt.plot(x, y)
    plt.show()


def perform_on(X, Y):
    embed = C[X]
    H = torch.tanh(embed.view(-1, block_size * dime_) @ W1 + B1)
    logits = H @ W2 + B2
    loss = F.cross_entropy(logits, Y)
    return loss


def plot_embedings():
    plt.figure(figsize=(10, 10))
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)

    for i in range(C.shape[0]):
        try:
            plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color="black")
        except KeyError:
            continue
      
    plt.grid(True)
    plt.show()

#------------------------------------------------------------------------#


def generate_text(num_words: int = 30, max_word_len: int = 20):
    """
    Generate a list of words character by character using the 
    trained weights.

    """
    generated_words = []
    
    for _ in range(num_words):
        out_word = []
        # Initialize the starting context with padding tokens (all '.' represented by 0)
        context = [0] * block_size
        
        while True:
            # Forward pass over a single input sequence
            x_input = torch.tensor([context]) # Shape: (1, block_size)
            embed = C[x_input]
            H = torch.tanh(embed.view(-1, block_size * dime_) @ W1 + B1)
            logits = H @ W2 + B2
            
            # Convert raw model scores into stable probabilities
            probs = F.softmax(logits, dim=1)
            
            # Sample the next character index based on the predicted probability distribution
            next_char_idx = torch.multinomial(probs, num_samples=1, generator=gen).item()
            
            # If the model samples a '.', the current generated word is complete
            if next_char_idx == 0:
                break
                
            out_word.append(itos[next_char_idx])
            
            # Crop the oldest token and append the newly predicted character index
            context = context[1:] + [next_char_idx]
            
            # Prevent infinite loops if the model gets stuck in a repetitive sequence
            if len(out_word) > max_word_len:
                break
                
        generated_words.append(''.join(out_word))
        
    return generated_words



#------------------------------------------------------------------------#

def save_weights(filepath: str = "virtual_usb_drive.pt"):
    """
    Serialize and save all trained network parameters and vocabulary mappings to a file.

    Args:
        filepath (str): Destination file path representing the virtual USB drive storage.
    """
    weights_dict = {
        "C": C.data,
        "W1": W1.data,
        "B1": B1.data,
        "W2": W2.data,
        "B2": B2.data,
        "stoi": stoi,
        "itos": itos
    }
    torch.save(weights_dict, filepath)
    print(f"💾 Success: Model weights compiled and exported to '{filepath}'!")



#---------------------------------------------------------------------------#

def load_weights(filepath: str = "virtual_usb_drive.pt"):
    """
    Load serialized parameters from a file and inject them back into the active global network.

    Args:
        filepath (str): Source file path representing the virtual USB drive storage.
    """
    global C, W1, B1, W2, B2, stoi, itos, vocab_size, parameters
    
    # Extract structural dictionary from the binary file
    weights_dict = torch.load(filepath)
    
    C = weights_dict["C"]
    W1 = weights_dict["W1"]
    B1 = weights_dict["B1"]
    W2 = weights_dict["W2"]
    B2 = weights_dict["B2"]
    stoi = weights_dict["stoi"]
    itos = weights_dict["itos"]
    vocab_size = len(stoi)
    
    # Re-assemble parameters list and restore gradients requirements tracking
    parameters = [C, W1, W2, B1, B2]
    for p in parameters:
        p.requires_grad = True
        
    print(f"🔌 Success: Model states loaded. Ready to decode from binary footprint!")
