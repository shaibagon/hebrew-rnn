from torch import nn


class LanguageModel(nn.Module):
    def __init__(self, dictionary_size, rec_type, hidden_size, num_layers):
        super(LanguageModel, self).__init__()
        # what type of recurrent layer
        rec_module = getattr(nn, rec_type)
        self.embed = nn.Embedding(num_embeddings=dictionary_size, embedding_dim=hidden_size)
        self.rec = rec_module(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=True, batch_first=False, dropout=0.5, bidirectional=False)
        # final output layer
        self.fc = nn.Linear(in_features=hidden_size, out_features=dictionary_size, bias=False)

    def forward(self, x, hidden):
        # embed the input x (should be in torch.Long format
        x = self.embed(x)
        if hidden is None:
            h_t, hidden = self.rec(x)
        else:
            h_t, hidden = self.rec(x, hidden)
        y = self.fc(h_t)
        return y, hidden