'''
Source code: https://docs.pytorch.org/docs/2.12/generated/torch.nn.GRU.html
'''

from local_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Method_GRU(method, nn.Module):
    data = None

    #hyperparameters
    max_epochs = 200
    learning_rate = 3e-4
    batch_size = 64

    weight_decay = 1e-4 #weight decay

    #GRU architecture settings
    embedding_dim = 64 #size of the dense vector each token is mapped to before entering GRU
    hidden_size = 128 #number of features in GRU hidden state
    num_layers = 1 #number of stacked GRU layers

    max_gen_length = 80 #maximum number of words to generate
    temperature = 0.7
    top_k = 20

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

        self.dropout = nn.Dropout(0.3)

        #GRU layer: processes embedded sequence step by step
        self.gru = nn.GRU(
            input_size=self.embedding_dim, #feature size per token
            hidden_size=self.hidden_size, #number of hidden units
            num_layers=self.num_layers, #stacked GRU depth
            batch_first=True, #tensors are (batch, seq, feature) so that it is easier to work with
            dropout=0.0 #dropout between layers only
        )

        #fully connected output layer: maps final hidden state to vocab logits
        self.fc = nn.Linear(self.hidden_size, vocab_size)

    #Forward Propagation
    def forward(self, x):
        embedded = self.embedding(x) #embed integers into dense vectors

        embedded = self.dropout(embedded)

        output, h_n = self.gru(embedded) #pass through GRU

        last_output = output[:, -1, :] #taking only last time step's output for the next-word prediction

        last_output = self.dropout(last_output)

        logits = self.fc(last_output) #project to vocabulary logits

        return logits

    #Training
    def train_model(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)  # ADDED label smoothing

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
                logits = self.forward(X_batch) #forward pass

                loss = loss_function(logits, y_batch) #compute loss

                #backward pass and parameter update
                optimizer.zero_grad() #clear gradients
                loss.backward() #compute loss by backpropagation

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) #gradient clipping

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

            # Filter UNK from starting words
            token_indices = []
            for w in starting_words:
                idx = vocab.get(w, None)
                if idx is not None and idx != unk_idx and idx != pad_idx:
                    token_indices.append(idx)
            if not token_indices:
                token_indices = [vocab.get('the', 2)]

            #pad left if the starting words are fewer than sequence_length
            if len(token_indices) < self.sequence_length:
                padding = [pad_idx] * (self.sequence_length - len(token_indices))
                token_indices = padding + token_indices

            context = token_indices[-self.sequence_length:] #keeping only last sequence_length tokens as initial context window

            generated_indices = [idx for idx in token_indices if idx not in (pad_idx, unk_idx)] #tracks all generated token indices

            for _ in range(self.max_gen_length):
                x = torch.LongTensor([context]) #building input tensor from current context window

                logits = self.forward(x) #forward pass -> logits over vocabulary

                #next_idx = logits.argmax(dim=1).item() #pick word with the highest probability (greedy decoding)

                logits = logits / self.temperature
                k = min(self.top_k, logits.size(1))
                top_k_logits, top_k_indices = torch.topk(logits[0], k)

                # Filter out UNK and PAD tokens
                valid_mask = (top_k_indices != unk_idx) & (top_k_indices != pad_idx)
                valid_indices = top_k_indices[valid_mask]
                valid_logits = top_k_logits[valid_mask]

                if len(valid_indices) == 0:
                    break

                probs = torch.softmax(valid_logits, dim=0)
                next_idx_from_valid = torch.multinomial(probs, num_samples=1).item()
                next_idx = valid_indices[next_idx_from_valid].item()

                if next_idx == eos_idx: #stop generation when EOS token is predicted
                    break

                generated_indices.append(next_idx)

                context = context[1:] + [next_idx] #sliding context window forward by 1

            #converts all indices back to words, skipping special tokens
        generated_words = []
        for idx in generated_indices:
            if idx not in (pad_idx, eos_idx, unk_idx):
                word = vocab_reverse.get(idx)
                if word and word not in ('<PAD>', '<UNK>', '<EOS>'):
                    generated_words.append(word)

        return ' '.join(generated_words)

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