from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(2)
torch.manual_seed(2)

# ---------------- LOAD DATA ----------------
data_obj = Dataset_Loader()
data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
data_obj.dataset_source_train_file_name = 'train.csv'
data_obj.dataset_source_test_file_name = 'test.csv'

data = data_obj.load()

# ---------------- MODEL ----------------
model = Method_MLP('MLP', '')
model.data = {
    'train': {'X': data['X_train'], 'y': data['y_train']},
    'test': {'X': data['X_test'], 'y': data['y_test']}
}

# ---------------- TRAIN/TEST ----------------
results = model.run()

# ---------------- SAVE ----------------
save_dir = '../../result/stage_2_result/'

saver = Result_Saver()
saver.result_destination_folder_path = save_dir
saver.result_destination_file_name = 'predictions'
saver.fold_count = 1
saver.data = results['pred_y']

print("Saving to:", save_dir + 'predictions_1.csv')

saver.save()

# ---------------- EVALUATION ----------------
evaluator = Evaluate_Accuracy()
evaluator.data = {
    'true_y': results['true_y'],
    'pred_y': results['pred_y']
}

acc = evaluator.evaluate()

print("\n************ Overall Performance ************")
print("Accuracy:", acc)

y_true = results['true_y']
y_pred = results['pred_y']

for avg in ['macro', 'weighted', 'micro']:
    print(f"\n{avg.upper()}")
    print("F1:", f1_score(y_true, y_pred, average=avg))
    print("Precision:", precision_score(y_true, y_pred, average=avg))
    print("Recall:", recall_score(y_true, y_pred, average=avg))

# ---------------- LOSS PLOT ----------------
plt.figure()
plt.plot(model.historical_loss)  
plt.title("Loss Convergence Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.savefig('../../result/stage_2_result/loss_convergence.png')

plt.show()
