from local_code.stage_4_code.text_generation.Jeslyn.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.text_generation.Jeslyn.Method_GRU import Method_GRU
from local_code.stage_4_code.text_generation.Jeslyn.Result_Saver import Result_Saver
import numpy as np
import torch
import matplotlib.pyplot as plt

if 1:
    np.random.seed(0)
    torch.manual_seed(0)

    data_obj = Dataset_Loader('joke generation', '')
    data_obj.dataset_source_folder_path = '../../../../data/stage_4_data/text_generation/'
    data_obj.dataset_source_file_name = 'data'

    load_data = data_obj.load() #load dataset and build vocab

    method_obj = Method_GRU(
        'gru text generation', '',
        vocab_size=load_data['vocab_size'],
        sequence_length=data_obj.sequence_length
    )

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../../../result/stage_4_result/Jeslyn/'
    result_obj.result_destination_file_name = 'generated_gru_'

    #starting words for generation
    starting_words = [
        #in dataset
        ['what', 'did', 'the'], #common
        ['why', 'did', 'the'], #common
        ['two', 'artists', 'had'], #not as common
        ['dont', 'you', 'hate'], #not as common
        ['i', 'like', 'my'], #not as common
        ['what', 'do', 'you'], #common
        #not in dataset
        ['how', 'do', 'baby'],
        ['why', 'did', 'pizza']
    ]

    method_obj.data = {
        'train':{
            'X': load_data['X'],
            'y': load_data['y']
        },
        'vocab': load_data['vocab'],
        'vocab_reverse': load_data['vocab_reverse'],
        'eos_idx': load_data['eos_idx'],
        'starting_words': starting_words
    }

    print('************ Start ************')
    results = method_obj.run()

    result_obj.data = results
    result_obj.save()

    plt.figure()
    plt.plot(results['train_losses'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GRU Text Generation Loss Curve')
    plt.savefig(
        result_obj.result_destination_folder_path +
        result_obj.result_destination_file_name +
        'text_generation_GRU_learning_curve.png'
    )
    plt.show()

    print('************ Finish ************')