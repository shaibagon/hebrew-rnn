import random
import string
import sys
from argparse import ArgumentParser

import torch
import torch.nn.functional as nnf
import tqdm
from torch.utils.data import IterableDataset

import hebrew
from model import LanguageModel

rstr = ''.join(random.choice(string.digits + string.ascii_lowercase) for _ in range(6))
parser = ArgumentParser()

# defining the model
parser.add_argument('--rec-type',   type=str,   default='RNN',help='type of reccurent module (RNN/LSTM/GRU...)')
parser.add_argument('--hidden',     type=int,   default=100,  help='number of hidden neurons in each layer')
parser.add_argument('--num-layers', type=int,   default=1,    help='number of hiddne layers in the model')
# defining the data
parser.add_argument('--seq-len',    type=int,   default=256,  help='training sequence length (length of each mini batch')
parser.add_argument('--batch',      type=int,   default=1024, help='batch size (number of concurrent sequences in each mini batch)')
# optimizer
parser.add_argument('--lr',         type=float, default=0.5,  help='learning rate for Adagrad optimizer')
# number of training epochs
parser.add_argument('--epochs',     type=int,   default=100,  help='number of training epochs')
parser.add_argument('--tag',        type=str,   default=rstr, help='unique tag to distinguish this run')
# warm start from checkpoint
parser.add_argument('--warm',       type=str,   default=None, help='path to checkpoint to wrm-start from')

args = parser.parse_args()


class BatchedSequence(IterableDataset):
    def __init__(self, sequence, batch_size, seq_len):
        super(BatchedSequence, self).__init__()
        self.seq = torch.from_numpy(sequence).to(dtype=torch.long)
        self.n = len(self.seq)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.overlap = 3  # relative overlap between batched sequences
        self.indices = None
        self.inputs_offset = torch.arange(0, self.seq_len, dtype=torch.long)[:, None]  # txB
        self.restart()

    def restart(self):
        # split the sequence across the batches - no randomness here
        self.indices = torch.arange(self.batch_size, dtype=torch.long)[None, :] * max(self.n // self.batch_size // self.overlap, 1)
        # # restart with random indices
        # self.indices = torch.randint(low=0, high=self.seq.shape[0]-self.seq_len-1,
        #                              size=(1, self.batch_size), dtype=torch.long)  # txB
        # self.indices[0, 0] = 0  # start from beginning every time.

        # make sure the last index is the most advance
        assert(torch.all(self.indices <= self.indices[0, -1]))

    def __iter__(self):
        # make sure the last index is the most advance
        assert(torch.all(self.indices <= self.indices[0, -1]))

        while self.indices[0, -1] < self.n - 1:  # no point training on the last character
            source_idx = self.indices + self.inputs_offset
            invalid = source_idx >= self.n - 1
            source_idx[invalid] = self.n - 2
            x = self.seq[source_idx]
            y = self.seq[source_idx + 1]
            y[invalid] = -1  # ignore these targets
            yield x, y
            self.indices += self.seq_len


def eval(model, dictionary, tempreture=1.0, epoch=-1):
    model.eval()
    with torch.no_grad():
        code = [torch.randint(low=0, high=dictionary.shape[0], size=(1, 1),
                              dtype=torch.long, device=torch.device('cuda'))]
        hidden = None
        for i in range(100):
            pred, hidden = model(code[-1].view(1, 1), hidden)
            # sample from the predicted probability
            prob = nnf.softmax(pred / tempreture, dim=-1)
            code.append(torch.multinomial(prob.flatten(), num_samples=1))
            # code.append(torch.argmax(pred, dim=-1))
        # convert code to simple list
        code = [c_.item() for c_ in code]
        print(f'eval {args.tag} ({epoch})=||{hebrew.code_to_text(code, dictionary)}||')
    model.train()


def main():
    # read the text
    bible = hebrew.read_bible()
    # convert it to code + dictionary
    code, dictionary = hebrew.convert_utf8_to_tokens(bible)

    model = LanguageModel(dictionary_size=len(dictionary), rec_type=args.rec_type,
                          hidden_size=args.hidden, num_layers=args.num_layers)
    model.cuda()
    opt = torch.optim.Adagrad(params=model.parameters(), lr=args.lr, weight_decay=0)

    # warm start model and optimizer
    if args.warm is not None:
        cp = torch.load(args.warm)
        print(f'Warm-start from {args.warm}. using checkpoint\'s args for model')
        assert((dictionary == cp['dictionary']).all())
        model = LanguageModel(dictionary_size=len(dictionary),
                              hidden_size=cp['args'].hidden, num_layers=cp['args'].num_layers)
        model.load_state_dict(cp['sd'])
        model.cuda()
        opt = torch.optim.Adagrad(params=model.parameters(), lr=args.lr, weight_decay=0)
        opt.load_state_dict(cp['opt'])

    # init
    eval(model, dictionary, tempreture=0.1, epoch=-1)

    data = BatchedSequence(code, seq_len=args.seq_len, batch_size=args.batch)

    tloss = 0
    for epoch in range(args.epochs):
        data.restart()
        hidden = None  # start fresh
        pbar = tqdm.tqdm(data, file=sys.stdout,
                         desc=f'train {args.tag} ({epoch}) loss={tloss:.2f}')
        for i, (x, y) in enumerate(pbar):
            x = x.cuda()
            y = y.cuda(non_blocking=True)  # y txB
            pred, hidden = model(x, hidden)  # pred txBxC
            loss = nnf.cross_entropy(pred.permute(0, 2, 1), y, ignore_index=-1)  # CE expects the "prob" dimension to be second
            opt.zero_grad()
            loss.backward()
            # clip gradients to range [-1, 1]
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.)
            opt.step()
            # stop gradients at hidden states
            if isinstance(hidden, torch.Tensor):
                hidden.detach_()
            else:
                for h_ in hidden:
                    h_.detach_()
            rm = min(i/(i+1), 0.99)  # running mean capped at window size of 100
            tloss = rm * tloss + (1-rm) * loss.item()
            pbar.set_description(desc=f'train {args.tag} ({epoch}) loss={tloss:.2f}')
        pbar.close()
        eval(model, dictionary, tempreture=0.1, epoch=epoch)
        if ((epoch + 1) % 50) == 0:
            torch.save({'sd': model.state_dict(), 'opt': opt.state_dict(),
                        'dictionary': dictionary, 'args': args},
                       f'{args.tag}-checkpoint-{epoch:05d}.pth.tar')


if __name__ == '__main__':
    main()
