from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


def print_all_metrics(y_true, y_pred):

    print("========== MICRO ==========")
    print("Precision:", precision_score(y_true, y_pred, average="micro", zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, average="micro", zero_division=0))
    print("F1:", f1_score(y_true, y_pred, average="micro", zero_division=0))

    print("========== MACRO ==========")
    print("Precision:", precision_score(y_true, y_pred, average="macro", zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, average="macro", zero_division=0))
    print("F1:", f1_score(y_true, y_pred, average="macro", zero_division=0))

    print("======== WEIGHTED ========")
    print("Precision:", precision_score(y_true, y_pred, average="weighted", zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, average="weighted", zero_division=0))
    print("F1:", f1_score(y_true, y_pred, average="weighted", zero_division=0))

if __name__ == "__main__":

    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('text_classification', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/'

    method_obj = Method_RNN(
        vocab_size=1,
        embed_dim=128,
        hidden_dim=128,
        mName='RNN',
        mDescription=''
    )

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    print('************ Start ************')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    learned_result = setting_obj.load_run_save_evaluate()[1]

    y_true = np.array(learned_result['true_y']).reshape(-1)
    y_pred = np.array(learned_result['pred_y']).reshape(-1)

    history = learned_result["history"]

    plt.figure()
    plt.plot(history["epoch"], history["loss"])
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.figure()
    plt.plot(history["epoch"], history["acc"])
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    plt.figure()
    plt.plot(history["epoch"], history["f1"])
    plt.title("F1 Curve")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.show()

    print('************ Overall Performance ************')
    print_all_metrics(y_true, y_pred)

    print('************ Finish ************')