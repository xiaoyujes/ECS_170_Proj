from local_code.base_class.setting import setting
import torch.nn as nn


class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):

        print('loading Stage 4 dataset and preparing split...')

        train_loader, test_loader, vocab = self.dataset.load()

        self.method.vocab_size = len(vocab)

        self.method.embedding = nn.Embedding(
            self.method.vocab_size,
            self.method.embed_dim
        )

        self.method.data = {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'vocab': vocab
        }

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        score = self.evaluate.evaluate()

        return score, learned_result