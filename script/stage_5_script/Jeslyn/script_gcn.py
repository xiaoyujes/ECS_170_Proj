from local_code.stage_5_code.Jeslyn.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Jeslyn.Method_GCN import Method_GCN
from local_code.stage_5_code.Jeslyn.Result_Saver_GCN import Result_Saver_GCN
from local_code.stage_5_code.Jeslyn.Evaluate_Accuracy_GCN import Evaluate_Accuracy_GCN
from sklearn.metrics import f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import numpy as np
import torch

# Hyperparameters tuned for each dataset
hyperparams_config = {
    'cora': {
        'hidden_size': 128,
        'dropout_rate': 0.5,
        'learning_rate': 1e-2,
        'weight_decay': 5e-4,
        'patience': 50,
        'max_epoch': 400,
        'num_layers': 2
    },
    'pubmed': {
        'hidden_size': 128,
        'dropout_rate': 0.5,
        'learning_rate': 5e-3,
        'weight_decay': 0,
        'patience': 50,
        'max_epoch': 400,
        'num_layers': 2
    },
    'citeseer': {
        'hidden_size': 64,
        'dropout_rate': 0.6,
        'learning_rate': 3e-3,
        'weight_decay': 1e-3,
        'patience': 40,
        'max_epoch': 300,
        'num_layers': 2
    }
}

# Function to train and evaluate with specific hyperparameters
def train_and_evaluate(dataset_name, data_path, hyperparams):
    print(f"\n{'=' * 80}")
    print(f"Training GCN on {dataset_name.upper()} Dataset")
    print(f"Architecture: {hyperparams['num_layers']}-layer GCN")
    print(f"Hyperparameters: hidden_size={hyperparams['hidden_size']}, "
          f"dropout={hyperparams['dropout_rate']}, "
          f"lr={hyperparams['learning_rate']}, "
          f"weight_decay={hyperparams['weight_decay']}, "
          f"patience={hyperparams['patience']}")
    print(f"{'=' * 80}")

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Load dataset
    data_obj = Dataset_Loader(f'Stage 5 {dataset_name}', '')
    data_obj.dataset_name = dataset_name
    data_obj.dataset_source_folder_path = data_path
    load_data = data_obj.load()

    # Initialize GCN method with custom hyperparameters
    method_obj = Method_GCN(f'GCN for {dataset_name}', '')
    method_obj.data = load_data

    # Apply hyperparameters
    method_obj.max_epoch = hyperparams.get('max_epoch', 300)
    method_obj.learning_rate = hyperparams['learning_rate']
    method_obj.weight_decay = hyperparams['weight_decay']
    method_obj.hidden_size = hyperparams['hidden_size']
    method_obj.dropout_rate = hyperparams['dropout_rate']
    method_obj.patience = hyperparams['patience']
    method_obj.num_layers = hyperparams['num_layers']

    # Run training and testing
    results = method_obj.run()

    # Save predictions
    result_obj = Result_Saver_GCN('saver', '')
    result_obj.result_destination_folder_path = '../../../result/stage_5_result/Jeslyn/'
    result_obj.result_destination_file_name = f'{dataset_name.capitalize()}_GCN_prediction_result'
    result_obj.dataset_name = dataset_name
    result_obj.fold_count = 1
    result_obj.data = {'predictions': results['pred_y']}
    result_obj.save()

    # Evaluate accuracy
    evaluate_obj = Evaluate_Accuracy_GCN('accuracy', '')
    evaluate_obj.data = {
        'true_y': results['true_y'],
        'pred_y': results['pred_y']
    }
    accuracy = evaluate_obj.evaluate()

    # Calculate additional metrics
    y_true = results['true_y']
    y_pred = results['pred_y']

    print(f'\n************ {dataset_name.upper()} Dataset Results ************')
    print(f'GCN Accuracy: {accuracy:.4f}')
    print(f'Test Loss: {results["test_loss"]:.4f}')

    avg_methods = ['weighted', 'macro', 'micro']
    metrics_summary = {}

    for avg in avg_methods:
        f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
        recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
        precision = precision_score(y_true, y_pred, average=avg, zero_division=0)

        metrics_summary[avg] = {'f1': f1, 'recall': recall, 'precision': precision}

        print(f'\n{avg.upper()} Average:')
        print(f'  F1-Score: {f1:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  Precision: {precision:.4f}')

    # Generate loss convergence plot (training + TEST loss)
    plt.figure(figsize=(10, 6))

    # Plot both training and TEST loss (changed from validation to test)
    epochs = range(1, len(results['train_loss_history']) + 1)
    plt.plot(epochs, results['train_loss_history'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, results['test_loss_history'], 'g-', label='Test Loss', linewidth=2)  # Changed to green for test loss

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Loss Convergence (Training vs Test) - {dataset_name.upper()} Dataset', fontsize=14)  # Updated title
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Mark the best TEST loss (changed from validation to test)
    best_epoch = np.argmin(results['test_loss_history'])
    best_loss = results['test_loss_history'][best_epoch]
    plt.plot(best_epoch + 1, best_loss, 'go', markersize=8, label=f'Best Test Loss: {best_loss:.4f}')

    plt.tight_layout()
    plt.savefig(f'../../../result/stage_5_result/Jeslyn/{dataset_name}_loss_convergence.png', dpi=150)
    plt.show()

    return accuracy, metrics_summary, results

# Data paths (adjust according to your setup)
data_paths = {
    'cora': '../../../data/stage_5_data/cora/',
    'pubmed': '../../../data/stage_5_data/pubmed/',
    'citeseer': '../../../data/stage_5_data/citeseer/'
}

# Run experiments for all datasets
all_results = {}

# Step 5-3: Train on Cora
all_results['cora'] = train_and_evaluate('cora', data_paths['cora'], hyperparams_config['cora'])

# Step 5-4: Train on Pubmed
all_results['pubmed'] = train_and_evaluate('pubmed', data_paths['pubmed'], hyperparams_config['pubmed'])

# Step 5-4: Train on Citeseer
all_results['citeseer'] = train_and_evaluate('citeseer', data_paths['citeseer'], hyperparams_config['citeseer'])

# Summary comparison
print("\n" + "=" * 80)
print("FINAL SUMMARY - COMPARISON OF ALL DATASETS")
print("=" * 80)
print(f"{'Dataset':<12} {'Test Acc':<10} {'Test Loss':<10} {'Best Test Loss':<12} {'Hidden Size':<10}")
print("-" * 60)

for dataset_name, (accuracy, _, results) in all_results.items():
    best_test_loss = min(results['test_loss_history'])  # Changed to test loss
    hidden_size = hyperparams_config[dataset_name]['hidden_size']
    print(
        f"{dataset_name:<12} {accuracy:.4f}     {results['test_loss']:.4f}     {best_test_loss:.4f}        {hidden_size:<10}")

print("\n" + "=" * 80)
print("All experiments completed successfully!")
print("Loss convergence plots (Training vs Test Loss) saved for all three datasets")
print("Results saved in: ../../../result/stage_5_result/Jeslyn/")
print("=" * 80)

# Optional: Create a combined TEST loss plot for comparison (changed from validation to test)
plt.figure(figsize=(12, 8))

for dataset_name, (_, _, results) in all_results.items():
    epochs = range(1, len(results['test_loss_history']) + 1)
    plt.plot(epochs, results['test_loss_history'], linewidth=2, label=f'{dataset_name.upper()}')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Test Loss', fontsize=12)  # Changed to Test Loss
plt.title('Test Loss Comparison Across Datasets', fontsize=14)  # Updated title
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../../result/stage_5_result/Jeslyn/all_datasets_loss_comparison.png', dpi=150)
plt.show()