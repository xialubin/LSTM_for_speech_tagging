import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# torch.manual_seed(1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
WORD_EMBEDDING_DIM = 5
WORD_HIDDEN_DIM = 5
CHAR_EMBEDDING_DIM = 3
CHAR_HIDDEN_DIM = 3


class LSTMTagger(nn.Module):
    def __init__(self, word_embed_dim, char_embed_dim, word_hidden_dim, char_hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.word_hidden_dim = word_hidden_dim
        self.char_hidden_dim = char_hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, word_embed_dim)
        self.char_embeddings = nn.Embedding(vocab_size, char_embed_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.word_lstm = nn.LSTM(word_embed_dim, word_hidden_dim)
        self.char_lstm = nn.LSTM(char_embed_dim, char_hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(word_hidden_dim + char_hidden_dim, tagset_size)
        self.word_hidden = self.init_word_hidden()
        self.char_hidden = self.init_char_hidden()

    def init_word_hidden(self):
        # Before we've done anything, we don't have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.word_hidden_dim),
                torch.zeros(1, 1, self.word_hidden_dim))

    def init_char_hidden(self):
        return (torch.zeros(1, 1, self.char_hidden_dim),
                torch.zeros(1, 1, self.char_hidden_dim))

    def forward(self, sentence):
        word_embeds = self.word_embeddings(sentence)
        char_embeds = self.char_embeddings(sentence)
        word_lstm_out, self.word_hidden = self.word_lstm(word_embeds.view(len(sentence), 1, -1), self.word_hidden)
        char_lstm_out, self.char_hidden = self.char_lstm(char_embeds.view(len(sentence), 1, -1), self.char_hidden)
        lstm_out = torch.cat([word_lstm_out, char_lstm_out], dim=-1)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(WORD_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, WORD_HIDDEN_DIM, CHAR_HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model.forward(inputs)
    print(tag_scores)

for epoch in range(300):
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.word_hidden = model.init_word_hidden()
        model.char_hidden = model.init_char_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model.forward(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model.forward(inputs)

    print(tag_scores)
