from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(2)
torch.manual_seed(2)

# ---------------- LOAD DATA ----------------
data_obj = Dataset_Loader_MNIST()
data_obj.data_root = '../../data/stage_3_data/MNIST'

data = data_obj.load()

X_train = data['train']['X']
y_train = data['train']['y']
X_test  = data['test']['X']
y_test  = data['test']['y']

# ---------------- MODEL ----------------
model = Method_CNN_MNIST(
    num_classes = 10,
    lr          = 1e-3,
    batch_size  = 64,
    max_epoch   = 30,
)

model.train(X_train, y_train, X_test, y_test)

# ---------------- PREDICT ----------------
y_pred = model.predict(X_test)

# ---------------- SAVE ----------------
save_dir = '../../result/stage_3_result/'

saver = Result_Saver()
saver.result_destination_folder_path = save_dir
saver.result_destination_file_name   = 'mnist_predictions'
saver.data = {'true_y': y_test.tolist(), 'pred_y': y_pred.tolist()}
saver.save()

# ---------------- EVALUATION ----------------
evaluator = Evaluate_Accuracy()
evaluator.data = {'true_y': y_test, 'pred_y': y_pred}

result = evaluator.evaluate()

print("\n************ Overall Performance ************")
print(Evaluate_Accuracy.result_to_str(result))

y_true = y_test
for avg in ['macro', 'weighted', 'micro']:
    print(f"\n{avg.upper()}")
    print("F1:       ", f1_score(y_true, y_pred, average=avg))
    print("Precision:", precision_score(y_true, y_pred, average=avg))
    print("Recall:   ", recall_score(y_true, y_pred, average=avg))

# ---------------- LEARNING CURVES ----------------
epochs = range(1, model.max_epoch + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('MNIST CNN – Learning Curves', fontsize=14)

ax1.plot(epochs, model.train_loss_history, 'b-o', markersize=3, label='Train Loss')
ax1.plot(epochs, model.test_loss_history,  'r-s', markersize=3, label='Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_title('Loss Curve')
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, [a*100 for a in model.train_acc_history], 'b-o', markersize=3, label='Train Acc')
ax2.plot(epochs, [a*100 for a in model.test_acc_history],  'r-s', markersize=3, label='Test Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy Curve')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('../../result/stage_3_result/learning_curve_mnist.png', dpi=150)
plt.show()
