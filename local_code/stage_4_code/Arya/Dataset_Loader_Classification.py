'''
Concrete IO class for IMDB text classification dataset.
Reads .txt files from train/pos, train/neg, test/pos, test/neg directories.
Cleans text, builds vocabulary, and converts to padded integer sequences.
'''

import os
import re
import numpy as np
from collections import Counter


class Dataset_Loader_Classification:

    def __init__(self):
        self.data_root  = '../../data/stage_4_data/text_classification'
        self.max_vocab  = 10000   # keep top N words
        self.max_length = 200     # pad/truncate sequences to this length
        self.vocab      = None    # built during load()

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def clean_text(text):
        # lowercase
        text = text.lower()
        # remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # remove punctuation and digits
        text = re.sub(r'[^a-z\s]', ' ', text)
        # collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # simple stopword removal
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'in', 'on',
            'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was',
            'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'shall', 'can', 'it', 'its', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'she', 'they', 'them', 'his', 'her', 'their', 'what',
            'which', 'who', 'not', 'no', 'so', 'up', 'out', 'as', 'into',
        }
        tokens = [w for w in text.split() if w not in stopwords and len(w) > 1]
        return tokens

    # ------------------------------------------------------------------
    # File reading
    # ------------------------------------------------------------------

    def _read_split(self, split):
        texts, labels = [], []
        for label_int, polarity in [(1, 'pos'), (0, 'neg')]:
            folder = os.path.join(self.data_root, split, polarity)
            for fname in os.listdir(folder):
                if not fname.endswith('.txt'):
                    continue
                with open(os.path.join(folder, fname),
                          'r', encoding='utf-8', errors='ignore') as f:
                    tokens = self.clean_text(f.read())
                texts.append(tokens)
                labels.append(label_int)
        return texts, labels

    # ------------------------------------------------------------------
    # Vocabulary + encoding
    # ------------------------------------------------------------------

    def _build_vocab(self, texts):
        counter = Counter(tok for doc in texts for tok in doc)
        most_common = counter.most_common(self.max_vocab - 2)
        # 0 = PAD, 1 = UNK
        self.vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}

    def _encode(self, texts):
        encoded = []
        for tokens in texts:
            ids = [self.vocab.get(t, 1) for t in tokens]  # 1 = UNK
            # truncate or pad to max_length
            if len(ids) >= self.max_length:
                ids = ids[:self.max_length]
            else:
                ids = ids + [0] * (self.max_length - len(ids))
            encoded.append(ids)
        return np.array(encoded, dtype=np.int64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self):
        print('[Classification] Reading train files ...')
        train_texts, train_labels = self._read_split('train')

        print('[Classification] Reading test files ...')
        test_texts,  test_labels  = self._read_split('test')

        print('[Classification] Building vocabulary ...')
        self._build_vocab(train_texts)   # vocab built on train only

        X_train = self._encode(train_texts)
        X_test  = self._encode(test_texts)

        y_train = np.array(train_labels, dtype=np.int64)
        y_test  = np.array(test_labels,  dtype=np.int64)

        print(f'  Vocab size : {len(self.vocab) + 2}')
        print(f'  Train      : {X_train.shape}')
        print(f'  Test       : {X_test.shape}')

        return {
            'train': {'X': X_train, 'y': y_train},
            'test':  {'X': X_test,  'y': y_test},
            'vocab_size': len(self.vocab) + 2,
        }
