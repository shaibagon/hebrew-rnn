import torch
import hebrew
from model import LanguageModel
import torch.nn.functional as nnf
from torch.utils.data import IterableDataset
import tqdm
import sys


class BatchedSequence(IterableDataset):
    def __init__(self, sequence, batch_size, seq_len):
        super(BatchedSequence, self).__init__()
        self.seq = torch.from_numpy(sequence).to(dtype=torch.long)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.indices = None
        self.inputs_offset = torch.arange(0, self.seq_len, dtype=torch.long)[:, None]  # txB
        self.target_offset = torch.arange(1, self.seq_len+1, dtype=torch.long)[:, None]
        self.restart()

    def restart(self):
        # restart with random indices
        self.indices = torch.randint(low=0, high=self.seq.shape[0]-self.seq_len-1,
                                     size=(1, self.batch_size), dtype=torch.long)  # txB
        self.indices[0, 0] = 0  # start from beginning every time.

    def __iter__(self):
        while torch.all(self.indices < len(self.seq) - self.seq_len):
            x = self.seq[self.indices + self.inputs_offset]
            y = self.seq[self.indices + self.target_offset]
            yield x, y
            self.indices += self.seq_len


def eval(model, dictionary, tempreture=1.0, epoch=-1):
    model.eval()
    with torch.no_grad():
        code = [torch.randint(low=0, high=dictionary.shape[0], size=(1, 1),
                              dtype=torch.long, device=torch.device('cuda'))]
        h, c = None, None
        for i in range(100):
            pred, h, c = model(code[-1].view(1, 1), h, c)
            # sample from the predicted probability
            prob = nnf.softmax(pred / tempreture, dim=-1)
            code.append(torch.multinomial(prob.flatten(), num_samples=1))
            # code.append(torch.argmax(pred, dim=-1))
        # convert code to simple list
        code = [c_.item() for c_ in code]
        print(f'eval ({epoch})=||{hebrew.code_to_text(code, dictionary)}||')
    model.train()


def main():
    # read the text
    bible = hebrew.read_bible()
    # convert it to code + dictionary
    code, dictionary = hebrew.convert_utf8_to_tokens(bible)

    model = LanguageModel(dictionary_size=len(dictionary), hidden_size=128, num_layers=3)
    model.cuda()

    # init
    eval(model, dictionary, tempreture=0.1, epoch=-1)

    data = BatchedSequence(code, seq_len=4096, batch_size=8)

    opt = torch.optim.SGD(params=model.parameters(), lr=0.5, momentum=0.9, weight_decay=0)

    tloss = 0
    for epoch in range(500):
        i = 0
        data.restart()
        h_0, c_0 = None, None  # start fresh
        pbar = tqdm.tqdm(data, file=sys.stdout,
                         desc=f'train ({epoch}) loss={tloss:.2f}')
        for x, y in pbar:
            x = x.cuda()
            y = y.cuda(non_blocking=True)  # y txB
            pred, h_0, c_0 = model(x, h_0, c_0)  # pred txBxC
            loss = nnf.cross_entropy(pred.permute(0, 2, 1), y)  # CE expects the "prob" dimension to be second
            opt.zero_grad()
            loss.backward()
            opt.step()
            # stop gradients at hidden states
            h_0.detach_()
            c_0.detach_()
            tloss = 0.99 * tloss + 0.01 * loss.item()
            i += 1
            pbar.set_description(desc=f'train ({epoch}) loss={tloss:.2f} i={i}')
        pbar.close()
        eval(model, dictionary, tempreture=0.1, epoch=epoch)
    torch.save({'sd': model.state_dict(), 'opt': opt.state_dict()}, f'checkpoint-{epoch:05d}.pth.tar')


if __name__ == '__main__':
    main()
