from local_code.stage_3_code.Jeslyn.script_data_loader import Dataset_Loader
from local_code.stage_3_code.Jeslyn.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Jeslyn.Result_Saver_ORL import Result_Saver_ORL
from local_code.stage_3_code.Jeslyn.Evaluate_Accuracy_ORL import Evaluate_Accuracy_ORL
from sklearn.metrics import f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import numpy as np
import torch

if 1:
    np.random.seed(7)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('Stage 3 ORL', '')
    data_obj.dataset_source_folder_path = '../../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'ORL'
    load_data = data_obj.load()

    method_obj = Method_CNN_ORL('CNN for ORL', '')
    method_obj.data = {
        'train': {'X': load_data['X_train'], 'y': load_data['y_train']},
        'test': {'X': load_data['X_test'], 'y': load_data['y_test']}
    }

    results = method_obj.run()

    result_obj = Result_Saver_ORL('saver', '')
    result_obj.result_destination_folder_path = '../../../result/stage_3_result/Jeslyn/'
    result_obj.result_destination_file_name = 'ORL_CNN_prediction_result'
    result_obj.fold_count = 1
    result_obj.data = results['pred_y']
    result_obj.save()

    evaluate_obj = Evaluate_Accuracy_ORL('accuracy', '')

    print('************ Start ************')
    evaluate_obj.data = {
        'true_y': results['true_y'],
        'pred_y': results['pred_y']
    }
    metrics = evaluate_obj.evaluate()

    print('************ Overall Performance ************')
    print(f'CNN Accuracy: {metrics}')
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
    plt.title('Loss Convergence Plot - ORL')
    plt.grid(True)
    plt.savefig('../../../result/stage_3_result/Jeslyn/loss_convergence.png')
    plt.show()