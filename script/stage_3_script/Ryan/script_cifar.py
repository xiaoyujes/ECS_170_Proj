from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_Cifar10 import Method_Cifar10
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy

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

    datasets = [
        ("MNIST", 1, 10),
        ("ORL", 1, 40),
        ("CIFAR", 3, 10)
    ]

    for dataset_name, channels, classes in datasets:

        print(f"************ {dataset_name} Start ************")

        data_obj = Dataset_Loader(dataset_name, "")
        data_obj.dataset_source_folder_path = "../../data/stage_3_data/"
        data_obj.dataset_source_file_name = [dataset_name]

        method_obj = Method_Cifar10("CNN", "")
        method_obj.set_in_channels(channels)
        method_obj.set_num_classes(classes)

        result_obj = Result_Saver("saver", "")
        result_obj.result_destination_folder_path = "../../result/stage_3_result/"
        result_obj.result_destination_file_name = dataset_name

        setting_obj = Setting_Train_Test_Split("setting", "")
        evaluate_obj = Evaluate_Accuracy("eval", "")

        setting_obj.prepare(
            data_obj,
            method_obj,
            result_obj,
            evaluate_obj
        )

        score, learned_result = setting_obj.load_run_save_evaluate()

        y_true = learned_result["true_y"]
        y_pred = learned_result["pred_y"]
        history = learned_result["history"]

        plt.figure()
        plt.plot(history["epoch"], history["loss"])
        plt.title(f"{dataset_name} Loss")
        plt.show()

        plt.figure()
        plt.plot(history["epoch"], history["acc"])
        plt.title(f"{dataset_name} Accuracy")
        plt.show()

        plt.figure()
        plt.plot(history["epoch"], history["f1"])
        plt.title(f"{dataset_name} F1")
        plt.show()

        print_all_metrics(y_true, y_pred)

        print(f"************ {dataset_name} Finish ************")