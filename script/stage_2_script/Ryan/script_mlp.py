from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


def print_all_metrics(y_true, y_pred):

    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("========== MICRO ==========")
    print("Precision:", precision_micro)
    print("Recall:", recall_micro)
    print("F1:", f1_micro)

    print("========== MACRO ==========")
    print("Precision:", precision_macro)
    print("Recall:", recall_macro)
    print("F1:", f1_macro)

    print("======== WEIGHTED ========")
    print("Precision:", precision_weighted)
    print("Recall:", recall_weighted)
    print("F1:", f1_weighted)


if __name__ == "__main__":

    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('real', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_source_file_name = ['train.csv', 'test.csv']

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
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
    plt.legend(["Loss"])
    plt.show()

    plt.figure()
    plt.plot(history["epoch"], history["acc"])
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy"])
    plt.show()

    plt.figure()
    plt.plot(history["epoch"], history["f1"])
    plt.title("F1 Curve")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend(["F1"])
    plt.show()

    print('************ Overall Performance ************')

    print_all_metrics(y_true, y_pred)

    print('************ Finish ************')