"""
Here leaves the clean code of the model.
"""
import core

class TextGenerator:

  def __init__(self, rawdata:str='data.md', block_size:int=3, 
               dime:int=2, **args):
    """ 
    All fundamental functions leave in the core.py 

    Initialize the required attributes:
    the training samples: input ---- > Xtr, output ---- > Ytr
    the dev samples:      input ---- > Xde, output ---- > Yde
    the test samples:     input ---- > Xte, output ---- > Yte   

    Args:
      rawdata (str): The file path of the training dataset.
      block_size (int): How many data tokens to take into account to predict the next?
      dime (int): Dimension of the embedding lookup table.
    """
    
    # 1. Store hyperparameters
    self.rawdata = rawdata
    self.block_size = block_size
    self.dime = dime 

    # Synchronize configuration variables in core.py with your choices
    core.block_size = self.block_size
    core.dime_ = self.dime

    self.Xtr, self.Ytr = None, None
    self.Xde, self.Yde = None, None 
    self.Xte, self.Yte = None, None
    self.embeding = None 

    # 2. Build the vocabulary mappings (stoi and itos)
    self.words, self.stoi, self.itos = core.build_vocab(self.rawdata)

    # 3. Network Initialization:
    # Pass the actual vocabulary size to core.py to dynamically generate 
    # the matrices C, W1, W2, B1, and B2.
    core.init_network(len(self.stoi))

    # 4. Retrieve the freshly initialized parameters from core.py
    self.params = core.parameters

    # Splitting datasets by index percentages
    self.d_height_per = int(.8 * len(self.words))
    self.d_nine_per =  int(.9 * len(self.words))


  def train(self, iter:int=200000):
    """ Train the model for a specified number of iterations. """
    # Build the training dataset split
    XTr, YTr = core.build_dataset(self.words[:self.d_height_per])
    
    # Run the training loop
    self.x, self.y = core.train(XTr, YTr, iter, 'lr')
    
    # Refresh local parameters reference after optimization updates
    self.params = core.parameters


  def show_training_eval(self, **args):
    """ Plot a graph representing the training loss/learning rate evaluation. """
    core.plot_fig(self.x, self.y)


  def perform_on(self, sets:str="dev", iter=200000):
    """ 
    Evaluate model performance on the validation dataset ('dev') 
    or the final evaluation dataset ('test') after training is done.
    """
    if sets.lower() not in ['dev', 'test']:
      raise ValueError("""
      Please note: the 'sets' argument must be 'dev' for the development dataset
      or 'test' for the test dataset.
      """)

    if sets.lower() == 'dev':
      words = self.words[self.d_height_per:self.d_nine_per]

      # Build the development dataset split
      XDe, YDe = core.build_dataset(words)

      # Evaluate and compute loss on dev split
      loss = core.perform_on(XDe, YDe)
      return loss

#-----------------------------------------------------------------------#
#   Perform on Test dataset split
#-----------------------------------------------------------------------#

    words = self.words[self.d_nine_per:]
    
    # Build the test dataset split
    XTe, YTe = core.build_dataset(words)
    
    # Evaluate and compute loss on test split
    loss = core.perform_on(XTe, YTe)
    return loss

    
  def num_parameters(self, *args):
    """ Return and print the total number of learnable parameters in the network. """
    print(sum(p.nelement() for p in core.parameters))


  def plot_embedings(self, *args):
    """ Visualize the learned 2D word/character embeddings using matplotlib. """
    core.plot_embedings()


  def generate(self, count: int = 30):
    """ 
    Decode and generate text sequences based on the trained model 
    parameters.
    
    Args:
        count (int): Total number of words to generate.
        
    Returns:
        str: A single string consisting of generated words separated by spaces.
        
    """
    words_list = core.generate_text(num_words=count)
    return " ".join(words_list)


  def save_weights(self, filename: str = "gnabro.pt"):
    """ 
    Trigger the core storage layer to export current parameter sets into a single file footprint.
    """
    core.save_weights(filename)


  def load_weights(self, filename: str = "gnabro.pt"):
    """ 
    Import weight states from a file and update local parameter pointers accordingly.
    """
    core.load_weights(filename)

    # Refresh local class state variables with new parameters mappings
    self.params = core.parameters
    self.stoi = core.stoi
    self.itos = core.itos
