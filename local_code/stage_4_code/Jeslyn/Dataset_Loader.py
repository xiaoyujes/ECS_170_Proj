'''
Source Code: 'Dataset_Loader.py' from '/stage_1_code'
'''

from local_code.base_class.dataset import dataset
import re
import csv

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    #Vocabulary special tokens
    PAD_TOKEN = '<PAD>' #index 0 for padding to create equal length
    UNK_TOKEN = '<UNK>' #index 1 for tokens not seen during training
    EOS_TOKEN = '<EOS>' #index 2 for marking the end of a joke, stopping generation

    def __init__(self, dName=None, dDescription=None, sequence_length=5):
        super().__init__(dName, dDescription)

        self.sequence_length = sequence_length
        self.vocab = None #converts word to index (used during loading and training)
        self.vocab_reverse = None #converts index back to word (used for text generation)

    #Text Cleaning
    def text_cleaning(self, text):
        text = text.lower() #converting text to lowercase
        text = re.sub(r'[^a-z0-9\s]', '', text) #removing text that isn't letters, digits, or spaces
        text = re.sub(r'\s+', ' ', text).strip() #removes instances of more than 1 whitespace character
        return text

    #Vocabulary Construction
    def build_vocab(self, tokenized_jokes):
        word_freq = {} #initializing empty dictionary

        #keeps track of the number of times each word appears across all jokes
        for tokens in tokenized_jokes:
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

        sorted_words = sorted(word_freq, key=lambda w: word_freq[w], reverse=True) #sorting by frequency in descending order

        vocab = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1, self.EOS_TOKEN: 2} #reserves index 0 for PAD and 1 for UNK

        #assigning index for each word
        for word in sorted_words:
            vocab[word] = len(vocab)

        vocab_reverse = {idx: word for word, idx in vocab.items()} #reversing vocab to find word where we have index

        return vocab, vocab_reverse

    #Sequence Construction (sliding window)
    def build_sequence(self, tokenized_jokes):
        X, y = [], [] #initializing empty lists to X and y

        seq_len = self.sequence_length

        for tokens in tokenized_jokes:
            indices = [self.vocab.get(t, self.vocab[self.UNK_TOKEN]) for t in tokens] #look up word t, but if not in vocab, return <UNK> instead

            if len(indices) < seq_len + 1: #skipping jokes that are too short to form an X, y pair
                continue

            for i in range(len(indices) - seq_len):
                X.append(indices[i:i + seq_len]) #taking slice of seq_len token indices starting at position i as input context
                y.append(indices[i + seq_len]) #takes single token immediately after window as target to predict

        return X, y

    #Loading model
    def load(self):
        print('loading data...')

        filepath = self.dataset_source_folder_path + self.dataset_source_file_name
        jokes_raw = []

        #reading file
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None) #skips the header
            print(f'Header: {header}') #sanity check

            #skips malformed rows
            for row in reader:
                if len(row) < 2:
                    continue

                joke_text = row[1].strip()

                if joke_text:
                    jokes_raw.append(joke_text)

        print(f'Total jokes loaded: {len(jokes_raw)}') #sanity check

        tokenized_jokes = [self.text_cleaning(j).split() + [self.EOS_TOKEN] for j in jokes_raw] #cleaning and tokenizing jokes (word by word)

        self.vocab, self.vocab_reverse = self.build_vocab(tokenized_jokes) #building both vocab and vocab reverse based on tokenized_jokes
        print(f'Vocabulary size: {len(self.vocab)} tokens (including {self.PAD_TOKEN}, {self.UNK_TOKEN}, and {self.EOS_TOKEN})')

        X,y = self.build_sequence(tokenized_jokes)
        print(f'Total (X,y) sequence pairs: {len(X)}')
        print(f'Sequence Length (context window): {self.sequence_length}')

        if X:
            sample_x = [self.vocab_reverse[idx] for idx in X[0]]
            sample_y = self.vocab_reverse[y[0]]
            print(f'Sample X[0] tokens: {sample_x}')
            print(f'Sample Y[0] tokens: {sample_y}')

        return {
            'X': X, #list of token-index sequence
            'y': y, #list of next-token indices
            'vocab': self.vocab, #word to index
            'vocab_reverse': self.vocab_reverse, #index to word
            'vocab_size': len(self.vocab),
            'eos_idx': self.vocab[self.EOS_TOKEN] #index of <EOS>
        }