'''
Source code: https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html
'''

from local_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Method_LSTM(method, nn.Module):
    data = None

    #hyperparameters
    max_epochs = 120
    learning_rate = 1e-4
    batch_size = 64

    #weight_decay = 1e-4 #weight decay

    #LSTM architecture settings
    embedding_dim = 64 #size of the dense vector each token is mapped to before entering LSTM
    hidden_size = 128 #number of features in LSTM hidden state
    num_layers = 1 #number of stacked LSTM layers

    max_gen_length = 50 #maximum number of words to generate
    #temperature = 0.6
    #top_k = 20

    def __init__(self, mName, mDescription, vocab_size, sequence_length):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        #embedding layer: converts integer token indices to dense vectors
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )

        #LSTM layer: processes embedded sequence step by step
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim, #feature size per token
            hidden_size=self.hidden_size, #number of hidden units
            num_layers=self.num_layers, #stacked LSTM depth
            batch_first=True, #tensors are (batch, seq, feature) so that it is easier to work with
            dropout=0.0 #dropout between layers only
        )

        self.layer_norm = nn.LayerNorm(self.hidden_size) #layer norm

        #self.dropout = nn.Dropout(0.2)

        #fully connected output layer: maps final hidden state to vocab logits
        self.fc = nn.Linear(self.hidden_size, vocab_size)

    #Forward Propagation
    def forward(self, x, hidden=None):
        embedded = self.embedding(x) #embed integers into dense vectors

        if hidden is not None:
            output, hidden = self.lstm(embedded, hidden)  #pass through hidden state
        else:
            output, hidden = self.lstm(embedded)  #initializing lstm
        last_output = output[:, -1, :]
        last_output = self.layer_norm(last_output)
        #last_output = self.dropout(last_output)
        logits = self.fc(last_output)
        return logits, hidden

    #Training
    def train_model(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate)

        loss_function = nn.CrossEntropyLoss()

        #dataset loader for mini-batch training
        X_tensor = torch.LongTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        train_losses = []

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for X_batch, y_batch in loader:
                #intializing hidden state
                h0 = torch.zeros(self.num_layers, X_batch.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, X_batch.size(0), self.hidden_size)

                logits, _ = self.forward(X_batch, (h0, c0)) #forward pass
                loss = loss_function(logits, y_batch) #compute loss

                #backward pass and parameter update
                optimizer.zero_grad() #clear gradients
                loss.backward() #compute loss by backpropagation
                #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) #gradient clipping (prevent exploding gradients)
                optimizer.step() #update weights

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {avg_loss}')

        return train_losses

    #Text Generation
    def generate(self, starting_words, vocab, vocab_reverse, eos_idx):
        self.eval() #switching to evaluation and disabling dropout

        with torch.no_grad():
            #converting starting words to indices
            unk_idx = vocab.get('<UNK>', 1)
            pad_idx = vocab.get('<PAD>', 0)

            # Filter UNK and PAD from starting words
            token_indices = []
            for w in starting_words:
                idx = vocab.get(w, None)
                if idx is not None and idx != unk_idx and idx != pad_idx:
                    token_indices.append(idx)
            if not token_indices:
                token_indices = [vocab.get('the', 2)]

            #initializing hidden states
            h = torch.zeros(self.num_layers, 1, self.hidden_size)
            c = torch.zeros(self.num_layers, 1, self.hidden_size)

            generated_words = [w for w in starting_words if w not in ('<UNK>', '<PAD>', '<EOS>')]  # Filter starting words
            context = token_indices.copy()

            #processing starting word to build context
            for token_idx in token_indices:
                x = torch.LongTensor([[token_idx]])
                _, (h, c) = self.lstm(self.embedding(x), (h, c))

            #generating tokens one by one
            for _ in range(self.max_gen_length):
                #using last generated token as input
                x = torch.LongTensor([[context[-1]]])
                embedded = self.embedding(x)
                lstm_out, (h, c) = self.lstm(embedded, (h, c))

                #logits = self.fc(self.dropout(lstm_out[:, -1, :])) #logits from last output
                #logits = self.fc(lstm_out[:, -1, :])
                last_output = self.layer_norm(lstm_out[:, -1, :])
                logits = self.fc(last_output)

                #logits = logits / self.temperature  #scale logits by temperature

                # Apply top-k filtering
                #k = min(self.top_k, logits.size(1))
                #top_k_logits, top_k_indices = torch.topk(logits[0], k)

                # Filter out UNK and PAD tokens
                #valid_mask = (top_k_indices != unk_idx) & (top_k_indices != pad_idx)
                #valid_indices = top_k_indices[valid_mask]
                #valid_logits = top_k_logits[valid_mask]

                #if len(valid_indices) == 0:
                    #break

                next_idx = logits.argmax(dim=1).item()

                #probs = torch.softmax(logits, dim=1)
                #next_idx = torch.multinomial(probs, 1).item()
                #next_idx_from_valid = torch.multinomial(probs, num_samples=1).item()
                #next_idx = valid_indices[next_idx_from_valid].item()

                if next_idx == eos_idx:
                    break

                #converting index to word
                next_word = vocab_reverse.get(next_idx, '<UNK>')
                if next_word not in ('<PAD>', '<UNK>', '<EOS>'):  # Filter special tokens
                    generated_words.append(next_word)
                context.append(next_idx)

                #managing length
                if len(context) > self.sequence_length:
                    context = context[-self.sequence_length:]

            # Final filter to ensure no special tokens remain
            filtered_words = [w for w in generated_words if w not in ('<PAD>', '<UNK>', '<EOS>')]

            if not filtered_words:
                return ' '.join([w for w in starting_words if w not in ('<PAD>', '<UNK>', '<EOS>')]) or "joke"

            return ' '.join(filtered_words)

    #Run
    def run(self):
        print('method running...')
        print('--start training...')
        train_losses = self.train_model(self.data['train']['X'], self.data['train']['y'])

        print('--start generating...')
        generated_samples = []
        for starting_words in self.data['starting_words']:
            generated_text = self.generate(
                starting_words,
                self.data['vocab'],
                self.data['vocab_reverse'],
                self.data['eos_idx']
            )
            print(f'Starting words : {starting_words}')
            print(f'Generated text : {generated_text}\n')
            generated_samples.append({
                'starting_words': ' '.join(starting_words),
                'generated': generated_text
            })

        return {
            'train_losses': train_losses,
            'generated_text': generated_samples
        }