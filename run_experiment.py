"""
S&P500 Early Classification Experiment Runner
Comprehensive experiment runner with detailed analysis and visualization
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report


def load_experiment_data(experiment_name, data_dir="thesis_datasets"):
    """Load train and test data for a specific experiment"""
    train_file = os.path.join(data_dir, f"{experiment_name}_TRAIN.ts")
    test_file = os.path.join(data_dir, f"{experiment_name}_TEST.ts")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Dataset files not found for {experiment_name}")
    
    print(f"Loading {experiment_name} dataset...")
    
    # Load training data
    X_train, y_train = load_from_tsfile_to_dataframe(train_file)
    X_train = np.asarray([[[v for v in channel] for channel in sample] for sample in X_train.to_numpy()])
    
    # Load test data
    X_test, y_test = load_from_tsfile_to_dataframe(test_file)
    X_test = np.asarray([[[v for v in channel] for channel in sample] for sample in X_test.to_numpy()])
    
    print(f"Loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Data shape: {X_train.shape} (samples, features, timesteps)")
    
    return X_train, y_train, X_test, y_test


def create_results_directory(experiment_name):
    """Create timestamped results directory with experiment name"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/{experiment_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def calculate_harmonic_mean(accuracy, earliness):
    """Calculate harmonic mean as specified in thesis"""
    if (1 - earliness) + accuracy == 0:
        return 0
    return (2 * (1 - earliness) * accuracy) / ((1 - earliness) + accuracy)


def plot_confusion_matrix(y_true, y_pred, results_dir, experiment_name):
    """Create and save confusion matrix plot"""
    # Labels: 1.0 = UP, 2.0 = DOWN, so sklearn orders as [1.0, 2.0] = [UP, DOWN]
    cm = confusion_matrix(y_true, y_pred, labels=[1.0, 2.0])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['UP', 'DOWN'], yticklabels=['UP', 'DOWN'])
    plt.title(f'Confusion Matrix - {experiment_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def plot_earliness_distribution(stop_timestamps, results_dir):
    """Plot distribution of guesses by earliness"""
    max_length = max(stop_timestamps) if stop_timestamps else 390
    earliness_values = [ts / max_length for ts in stop_timestamps]
    
    plt.figure(figsize=(10, 6))
    plt.hist(earliness_values, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Earliness (fraction of sequence observed)')
    plt.ylabel('Number of predictions')
    plt.title('Distribution of Predictions by Earliness')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'earliness_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_earliness_vs_accuracy(stop_timestamps, y_test, y_pred, results_dir):
    """Plot earliness vs accuracy for individual test sequences"""
    if not stop_timestamps:
        print("Warning: No stop timestamps provided, skipping earliness vs accuracy plot.")
        return
        
    max_length = max(stop_timestamps)
    earliness_values = [ts / max_length for ts in stop_timestamps]
    correct_predictions = [1 if y_test[i] == y_pred[i] else 0 for i in range(len(y_test))]
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with different colors for correct/incorrect
    correct_mask = np.array(correct_predictions) == 1
    plt.scatter(np.array(earliness_values)[correct_mask], 
               np.array(correct_predictions)[correct_mask], 
               alpha=0.6, color='green', label='Correct', s=30)
    plt.scatter(np.array(earliness_values)[~correct_mask], 
               np.array(correct_predictions)[~correct_mask], 
               alpha=0.6, color='red', label='Incorrect', s=30)
    
    plt.xlabel('Earliness (fraction of sequence observed)')
    plt.ylabel('Prediction Correctness')
    plt.title('Earliness vs Accuracy for Individual Test Sequences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'earliness_vs_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_results_text(results_dict, results_dir):
    """Save all text results to file"""
    results_file = os.path.join(results_dir, 'results.txt')
    
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("S&P500 EARLY CLASSIFICATION EXPERIMENT RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("EXPERIMENT SETTINGS:\n")
        f.write(f"Experiment: {results_dict['experiment_name']}\n")
        f.write(f"CALIMERA Model: {results_dict['model_type']}\n")
        f.write(f"Delay Penalty: {results_dict['delay_penalty']}\n")
        f.write(f"Train Samples: {results_dict['train_samples']}\n")
        f.write(f"Test Samples: {results_dict['test_samples']}\n")
        f.write(f"Features: {results_dict['num_features']}\n")
        f.write(f"Sequence Length: {results_dict['sequence_length']}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"Average Accuracy: {results_dict['accuracy']:.4f}\n")
        f.write(f"Average Earliness: {results_dict['earliness']:.4f}\n")
        f.write(f"F1-Score: {results_dict['f1_score']:.4f}\n")
        f.write(f"Harmonic Mean: {results_dict['harmonic_mean']:.4f}\n")
        f.write(f"Cost: {results_dict['cost']:.4f}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        per_class = results_dict['per_class_metrics']
        f.write(f"UP   - Accuracy: {per_class['up_accuracy']:.4f} ({per_class['up_correct']}/{per_class['up_total']}) | Earliness: {per_class['up_earliness']:.4f}\n")
        f.write(f"DOWN - Accuracy: {per_class['down_accuracy']:.4f} ({per_class['down_correct']}/{per_class['down_total']}) | Earliness: {per_class['down_earliness']:.4f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        # Labels: 1.0 = UP, 2.0 = DOWN, so cm[0,0] = UP-UP, cm[1,1] = DOWN-DOWN
        cm = results_dict['confusion_matrix']
        f.write(f"         Predicted\n")
        f.write(f"          UP  DOWN\n")
        f.write(f"UP    {cm[0,0]:6d} {cm[0,1]:5d}\n")
        f.write(f"DOWN  {cm[1,0]:6d} {cm[1,1]:5d}\n\n")
        
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write(results_dict['classification_report'])
        f.write("\n")
        
        f.write("EARLINESS STATISTICS:\n")
        stop_times = results_dict['stop_timestamps']
        if stop_times:
            f.write(f"Mean Stop Time: {np.mean(stop_times):.2f}\n")
            f.write(f"Median Stop Time: {np.median(stop_times):.2f}\n")
            f.write(f"Min Stop Time: {min(stop_times)}\n")
            f.write(f"Max Stop Time: {max(stop_times)}\n")
            f.write(f"Std Dev Stop Time: {np.std(stop_times):.2f}\n")


def calculate_per_class_metrics(y_test, y_pred, stop_timestamps):
    """Calculate per-class accuracy and earliness"""
    y_test = np.array(y_test, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    stop_timestamps = np.array(stop_timestamps)
    
    # UP class (label 1.0)
    up_mask = y_test == 1.0
    up_correct = np.sum((y_test[up_mask] == y_pred[up_mask]))
    up_total = np.sum(up_mask)
    up_accuracy = up_correct / up_total if up_total > 0 else 0
    up_earliness = np.mean(stop_timestamps[up_mask]) / 390 if up_total > 0 else 0
    
    # DOWN class (label 2.0)
    down_mask = y_test == 2.0
    down_correct = np.sum((y_test[down_mask] == y_pred[down_mask]))
    down_total = np.sum(down_mask)
    down_accuracy = down_correct / down_total if down_total > 0 else 0
    down_earliness = np.mean(stop_timestamps[down_mask]) / 390 if down_total > 0 else 0
    
    return {
        'up_accuracy': up_accuracy, 'up_earliness': up_earliness, 
        'up_correct': up_correct, 'up_total': up_total,
        'down_accuracy': down_accuracy, 'down_earliness': down_earliness,
        'down_correct': down_correct, 'down_total': down_total
    }


def run_experiment():
    """Main experiment runner with user interaction"""
    print("=" * 60)
    print("S&P500 EARLY CLASSIFICATION EXPERIMENT RUNNER")
    print("=" * 60)
    
    # Available experiments
    experiments = [
        "SP500_ONLY",
        "SP500_VIX", 
        "SP500_VIX_MACRO",
        "SP500_VIX_TECH",
        "TECH_ONLY",
        "SP500_VIX_MACRO_TECH"
    ]
    
    # Show available experiments
    print("\nAvailable experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp}")
    
    # User selects experiment
    while True:
        try:
            choice = int(input("\nSelect experiment (1-6): "))
            if 1 <= choice <= 6:
                experiment_name = experiments[choice - 1]
                break
            else:
                print("Please enter a number between 1 and 6.")
        except ValueError:
            print("Please enter a valid number.")
    
    # User selects model type
    print("\nAvailable CALIMERA models:")
    print("1. Standard CALIMERA (calimera.py)")
    print("2. Weighted CALIMERA (calimera_weighted.py)")
    
    while True:
        try:
            model_choice = int(input("\nSelect model (1-2): "))
            if model_choice == 1:
                from calimera import CALIMERA
                model_type = "Standard CALIMERA"
                break
            elif model_choice == 2:
                from calimera_weighted import CALIMERA
                model_type = "Weighted CALIMERA"
                break
            else:
                print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")
        except ImportError as e:
            print(f"Error importing model: {e}")
            print("Make sure the file exists and is accessible.")
            return
    
    # User sets delay penalty
    while True:
        try:
            delay_penalty = float(input("\nEnter delay penalty (0.0-1.0, default 0.6): ") or "0.6")
            if 0.0 <= delay_penalty <= 1.0:
                break
            else:
                print("Delay penalty must be between 0.0 and 1.0.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"Model: {model_type}")
    print(f"Delay penalty: {delay_penalty}")
    print(f"{'='*60}")
    
    # Create results directory
    results_dir = create_results_directory(experiment_name)
    print(f"\nResults will be saved to: {results_dir}")
    
    try:
        # Load data
        X_train, y_train, X_test, y_test = load_experiment_data(experiment_name)
        
        # Train model
        print("\nTraining CALIMERA model...")
        model = CALIMERA(delay_penalty=delay_penalty)
        model.fit(X_train, y_train)
        
        # Test model
        print("Testing model...")
        stop_timestamps, y_pred = model.test(X_test)
        
        # Ensure y_test and y_pred are float arrays for consistent metric calculations
        y_test = np.array(y_test, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        # Calculate metrics
        print("\nCalculating metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        earliness = sum(stop_timestamps) / (X_test.shape[-1] * X_test.shape[0])
        f1 = f1_score(y_test, y_pred, average='weighted')
        harmonic_mean = calculate_harmonic_mean(accuracy, earliness)
        cost = 1.0 - accuracy + delay_penalty * earliness
        cm = confusion_matrix(y_test, y_pred, labels=[1.0, 2.0])  # UP, DOWN order
        class_report = classification_report(y_test, y_pred, target_names=['UP', 'DOWN'])
        
        # Calculate per-class metrics
        per_class_metrics = calculate_per_class_metrics(y_test, y_pred, stop_timestamps)
        
        # Prepare results dictionary
        results_dict = {
            'experiment_name': experiment_name,
            'model_type': model_type,
            'delay_penalty': delay_penalty,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'num_features': X_train.shape[1],
            'sequence_length': X_train.shape[2],
            'accuracy': accuracy,
            'earliness': earliness,
            'f1_score': f1,
            'harmonic_mean': harmonic_mean,
            'cost': cost,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'stop_timestamps': stop_timestamps,
            'per_class_metrics': per_class_metrics
        }
        
        # Print results
        print(f"\n{'='*40}")
        print("EXPERIMENT RESULTS")
        print(f"{'='*40}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Earliness: {earliness:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Harmonic Mean: {harmonic_mean:.4f}")
        print(f"Cost: {cost:.4f}")
        print("\nPer-Class Metrics:")
        print(f"UP   - Accuracy: {per_class_metrics['up_accuracy']:.4f} ({per_class_metrics['up_correct']}/{per_class_metrics['up_total']}) | Earliness: {per_class_metrics['up_earliness']:.4f}")
        print(f"DOWN - Accuracy: {per_class_metrics['down_accuracy']:.4f} ({per_class_metrics['down_correct']}/{per_class_metrics['down_total']}) | Earliness: {per_class_metrics['down_earliness']:.4f}")
        print("\nConfusion Matrix:")
        print(f"         Predicted")
        print(f"          UP  DOWN")
        print(f"UP    {cm[0,0]:6d} {cm[0,1]:5d}")
        print(f"DOWN  {cm[1,0]:6d} {cm[1,1]:5d}")
        
        # Print per-class metrics
        up_metrics = per_class_metrics['up_accuracy'], per_class_metrics['up_earliness']
        down_metrics = per_class_metrics['down_accuracy'], per_class_metrics['down_earliness']
        print(f"\nPer-Class Metrics:")
        print(f"UP    - Accuracy: {up_metrics[0]:.4f}, Earliness: {up_metrics[1]:.4f}")
        print(f"DOWN  - Accuracy: {down_metrics[0]:.4f}, Earliness: {down_metrics[1]:.4f}")
        
        # Create visualizations
        print(f"\nGenerating visualizations...")
        plot_confusion_matrix(y_test, y_pred, results_dir, experiment_name)
        plot_earliness_distribution(stop_timestamps, results_dir)
        plot_earliness_vs_accuracy(stop_timestamps, y_test, y_pred, results_dir)
        
        # Save text results
        print("Saving results...")
        save_results_text(results_dict, results_dir)
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {results_dir}")
        print(f"Generated plots: confusion_matrix.png, earliness_distribution.png, earliness_vs_accuracy.png")
        print(f"Text results: results.txt")
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        print("Please check that all required files exist and try again.")


def run_experiment_programmatic(experiment_name, model_type="standard", delay_penalty=0.6):
    """
    Run experiment programmatically without user input
    
    Args:
        experiment_name: One of ["SP500_ONLY", "SP500_VIX", "SP500_VIX_MACRO", 
                                "SP500_VIX_TECH", "TECH_ONLY", "SP500_VIX_MACRO_TECH"]
        model_type: "standard" or "weighted" 
        delay_penalty: float between 0.0 and 1.0
    """
    
    # Validate inputs
    valid_experiments = [
        "SP500_ONLY", "SP500_VIX", "SP500_VIX_MACRO",
        "SP500_VIX_TECH", "TECH_ONLY", "SP500_VIX_MACRO_TECH"
    ]
    
    if experiment_name not in valid_experiments:
        raise ValueError(f"Invalid experiment. Choose from: {valid_experiments}")
    
    if model_type not in ["standard", "weighted"]:
        raise ValueError("Model type must be 'standard' or 'weighted'")
    
    if not 0.0 <= delay_penalty <= 1.0:
        raise ValueError("Delay penalty must be between 0.0 and 1.0")
    
    # Import the appropriate model
    if model_type == "standard":
        from calimera import CALIMERA
        model_type_name = "Standard CALIMERA"
    else:
        from calimera_weighted import CALIMERA
        model_type_name = "Weighted CALIMERA"
    
    print(f"Running {experiment_name} with {model_type_name} (delay_penalty={delay_penalty})")
    
    # Create results directory
    results_dir = create_results_directory(experiment_name)
    print(f"Results will be saved to: {results_dir}")
    
    try:
        # Load data
        X_train, y_train, X_test, y_test = load_experiment_data(experiment_name)
        
        # Train model
        print("Training CALIMERA model...")
        model = CALIMERA(delay_penalty=delay_penalty)
        model.fit(X_train, y_train)
        
        # Test model
        print("Testing model...")
        stop_timestamps, y_pred = model.test(X_test)
        
        # Ensure y_test and y_pred are float arrays for consistent metric calculations
        y_test = np.array(y_test, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        # Calculate metrics
        print("Calculating metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        earliness = sum(stop_timestamps) / (X_test.shape[-1] * X_test.shape[0])
        f1 = f1_score(y_test, y_pred, average='weighted')
        harmonic_mean = calculate_harmonic_mean(accuracy, earliness)
        cost = 1.0 - accuracy + delay_penalty * earliness
        cm = confusion_matrix(y_test, y_pred, labels=[1.0, 2.0])  # UP, DOWN order
        class_report = classification_report(y_test, y_pred, target_names=['UP', 'DOWN'])
        
        # Calculate per-class metrics
        per_class_metrics = calculate_per_class_metrics(y_test, y_pred, stop_timestamps)
        
        # Prepare results dictionary
        results_dict = {
            'experiment_name': experiment_name,
            'model_type': model_type_name,
            'delay_penalty': delay_penalty,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'num_features': X_train.shape[1],
            'sequence_length': X_train.shape[2],
            'accuracy': accuracy,
            'earliness': earliness,
            'f1_score': f1,
            'harmonic_mean': harmonic_mean,
            'cost': cost,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'stop_timestamps': stop_timestamps,
            'per_class_metrics': per_class_metrics
        }
        
        # Print results
        print(f"\n{'='*40}")
        print("EXPERIMENT RESULTS")
        print(f"{'='*40}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Earliness: {earliness:.4f}")
        print("Earliness in min: {(earliness*390):.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Harmonic Mean: {harmonic_mean:.4f}")
        print(f"Cost: {cost:.4f}")
        print("\nPer-Class Metrics:")
        print(f"UP   - Accuracy: {per_class_metrics['up_accuracy']:.4f} ({per_class_metrics['up_correct']}/{per_class_metrics['up_total']}) | Earliness: {per_class_metrics['up_earliness']:.4f}")
        print(f"DOWN - Accuracy: {per_class_metrics['down_accuracy']:.4f} ({per_class_metrics['down_correct']}/{per_class_metrics['down_total']}) | Earliness: {per_class_metrics['down_earliness']:.4f}")
        
        # Create visualizations
        if HAS_PLOTTING:
            print("Generating visualizations...")
            plot_confusion_matrix(y_test, y_pred, results_dir, experiment_name)
            plot_earliness_distribution(stop_timestamps, results_dir)
            plot_earliness_vs_accuracy(stop_timestamps, y_test, y_pred, results_dir)
        
        # Save text results
        save_results_text(results_dict, results_dir)
        
        print(f"\nExperiment completed! Results saved to: {results_dir}")
        return results_dict, results_dir
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    # Check if running interactively or with command line arguments
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode: python run_experiment.py EXPERIMENT_NAME [MODEL_TYPE] [DELAY_PENALTY]
        experiment_name = sys.argv[1]
        model_type = sys.argv[2] if len(sys.argv) > 2 else "standard"
        delay_penalty = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
        
        print(f"Running in command line mode:")
        print(f"  Experiment: {experiment_name}")
        print(f"  Model: {model_type}")
        print(f"  Delay penalty: {delay_penalty}")
        
        run_experiment_programmatic(experiment_name, model_type, delay_penalty)
    else:
        # Interactive mode
        run_experiment()
