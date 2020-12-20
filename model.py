from torch import nn


class LanguageModel(nn.Module):
    def __init__(self, dictionary_size, hidden_size, num_layers):
        super(LanguageModel, self).__init__()
        self.embed = nn.Embedding(num_embeddings=dictionary_size, embedding_dim=hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                            bias=True, batch_first=False, dropout=0.1, bidirectional=False)
        # final output layer
        self.fc = nn.Linear(in_features=hidden_size, out_features=dictionary_size, bias=False)

    def forward(self, x, h_0, c_0):
        # embed the input x (should be in torch.Long format
        x = self.embed(x)
        if h_0 is None and c_0 is None:
            h_t, (h_n, c_n) = self.lstm(x)
        else:
            h_t, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        y = self.fc(h_t)
        return y, h_n, c_n