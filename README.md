## Hebrew RNN - Training a Hebrew Language Model using PyTorch
This is a toy project implementing a simple char-based languge model using deep LSTM model.

### Training data
In principle, the model can be trained on any large enough sequence of characters.  
This implementation used a copy of the bible in hebrew from [Machon Mamre](https://www.mechon-mamre.org/dlk.htm).
To use it for training the model, doenload and unzip it to subfolder `./k`.

### Training from command line
```python
usage: rnn.py [-h] [--hidden HIDDEN] [--num-layers NUM_LAYERS]
              [--seq-len SEQ_LEN] [--batch BATCH] [--lr LR] [--epochs EPOCHS]
              [--tag TAG] [--warm WARM]

optional arguments:
  -h, --help            show this help message and exit
  --hidden HIDDEN       number of hidden neurons in each layer
  --num-layers NUM_LAYERS
                        number of hiddne layers in the model
  --seq-len SEQ_LEN     training sequence length (length of each mini batch
  --batch BATCH         batch size (number of concurrent sequences in each
                        mini batch)
  --lr LR               learning rate for Adagrad optimizer
  --epochs EPOCHS       number of training epochs
  --tag TAG             unique tag to distinguish this run
  --warm WARM           path to checkpoint to warm-start from
```

### Play with a trained model on command line
```python
import torch
import torch.nn.functional as nnf
import hebrew
from model import LanguageModel

# load checkpoint
cp = torch.load('my-checkpoint.pth.tar')
dictionary = cp['dictionary']
# make model based on the saved one
model = LanguageModel(dictionary_size=len(dictionary),
                      hidden_size=cp['args'].hidden, num_layers=cp['args'].num_layers)
# load the weights
model.load_state_dict(cp['sd'])
model.eval()

temprature = 0.01  # higher temperature = more randomness in sampling from posterior

with torch.no_grad():
  # start generating from random 
  code = [torch.randint(low=0, high=dictionary.shape[0], size=(1, 1), dtype=torch.long)]
  h, c = None, None
  for i in range(100):
    pred, h, c = model(code[-1].view(1, 1), h, c)
    # sample from the predicted probability
    prob = nnf.softmax(pred / temprature, dim=-1)
    code.append(torch.multinomial(prob.flatten(), num_samples=1))
  # convert code to simple list
  code = [c_.item() for c_ in code]
  print(f'||{hebrew.code_to_text(code, dictionary)[::-1]}||')  # sometimes hebrew strings needs to be reversed for printing...
```
