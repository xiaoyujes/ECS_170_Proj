from local_code.stage_5_code.Dataset_Loader_CiteSeer import Dataset_Loader_CiteSeer
from local_code.stage_5_code.Method_GCN import Method_GCN
from local_code.stage_5_code.Result_Saver import Result_Saver
from local_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(2)
torch.manual_seed(2)

# ---------------- LOAD DATA ----------------
data_obj = Dataset_Loader_CiteSeer()
data_obj.dataset_source_folder_path = '../../data/stage_5_data/citeseer'

data = data_obj.load()

graph          = data['graph']
train_test_val = data['train_test_val']

# ---------------- MODEL ----------------
in_features = graph['X'].shape[1]

model = Method_GCN(
    in_features  = in_features,
    hidden_dim   = 64,
    num_classes  = 6,
    dropout      = 0.5,
    lr           = 1e-2,
    weight_decay = 5e-4,
    max_epoch    = 500,
)

model.train(graph, train_test_val)

# ---------------- PREDICT ----------------
idx_test = train_test_val['idx_test']
y_pred   = model.predict(graph, idx_test)
y_true   = graph['y'][idx_test].numpy()

# ---------------- SAVE ----------------
save_dir = '../../result/stage_5_result/'

saver = Result_Saver()
saver.result_destination_folder_path = save_dir
saver.result_destination_file_name   = 'citeseer_predictions'
saver.data = {'true_y': y_true.tolist(), 'pred_y': y_pred.tolist()}
saver.save()

# ---------------- EVALUATION ----------------
evaluator = Evaluate_Accuracy()
evaluator.data = {'true_y': y_true, 'pred_y': y_pred}
result = evaluator.evaluate()
print(Evaluate_Accuracy.result_to_str(result))

# ---------------- LEARNING CURVES ----------------
epochs = range(1, model.max_epoch + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('CiteSeer GCN - Learning Curves', fontsize=14)

ax1.plot(epochs, model.train_loss_history, 'b-', linewidth=1, label='Train Loss')
ax1.plot(epochs, model.test_loss_history,  'r-', linewidth=1, label='Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('NLL Loss')
ax1.set_title('Loss Curve')
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, [a*100 for a in model.train_acc_history], 'b-', linewidth=1, label='Train Acc')
ax2.plot(epochs, [a*100 for a in model.test_acc_history],  'r-', linewidth=1, label='Test Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy Curve')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('../../result/stage_5_result/learning_curve_citeseer.png', dpi=150)
plt.show()
