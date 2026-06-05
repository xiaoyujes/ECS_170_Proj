from local_code.stage_5_code.Method_GCN import Method_GCN


class Method_GCN_Pubmed(Method_GCN):
    """Best plain 2-layer GCN settings found for Pubmed."""

    def __init__(self, mName='pubmed best graph convolutional network', mDescription=''):
        super().__init__(mName, mDescription)
        self.max_epoch = 300
        self.learning_rate = 0.01
        self.weight_decay = 5e-4
        self.hidden_dim = 48
        self.hidden_dims = None
        self.dropout = 0.35
        self.use_best_validation_model = False
        self.best_selection_metric = 'accuracy'
        self.print_interval = 10
        self.early_stopping_patience = None
        self.use_validation = True
