from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
import torch
import matplotlib.pyplot as plt

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('Stage 2', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_source_train_file_name = 'train.csv'
    data_obj.dataset_source_test_file_name = 'test.csv'

    load_data = data_obj.load()

    method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj.data = {
        'train':{
            'X': load_data['X_train'],
            'y': load_data['y_train'],
        },
        'test':{
            'X': load_data['X_test'],
            'y': load_data['y_test'],
        }
    }

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/'
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = 1

    results = method_obj.run()

    result_obj.data = results['pred_y']
    result_obj.save()

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    evaluate_obj.data = {
        'true_y': results['true_y'],
        'pred_y': results['pred_y']
    }
    metrics = evaluate_obj.evaluate()

    print('************ Overall Performance ************')
    print(f'MLP Accuracy: {metrics}')
    print('Other Metrics: ')
    avg_methods = ['weighted', 'macro', 'micro']

    y_true = results['true_y']
    y_pred = results['pred_y']

    for avg in avg_methods:
        f1 = f1_score(y_true, y_pred, average=avg)
        recall = recall_score(y_true, y_pred, average=avg)
        precision = precision_score(y_true, y_pred, average=avg)

        print(f'F1-Score - {avg}: {f1}')
        print(f'Recall - {avg}: {recall}')
        print(f'Precision - {avg}: {precision}')
    # ------------------------------------------------------

    # --- Loss Convergence Plot -------
    plt.figure()
    plt.plot(method_obj.historical_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Convergence Plot')
    plt.grid(True)
    plt.savefig('../../result/stage_2_result/loss_convergence.png')
    plt.show()
    

    