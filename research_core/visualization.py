import matplotlib.pyplot as plt
import os

def plot_results(experiments_history, save_dir='results'):
    """
    Args:
        experiments_history (dict): Training history of each experiment.
        save_dir (str): Folder to save graphics.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    colors = plt.cm.tab20(range(len(experiments_history)))

    plt.figure(figsize=(12, 8))
    
    for i, (exp_name, history) in enumerate(experiments_history.items()):
  
        plt.plot(history['val_prauc'], label=exp_name, color=colors[i], linewidth=2)

    plt.title('Validation PR-AUC Comparison (Higher is Better)', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('PR-AUC Score', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = f"{save_dir}/comparison_prauc.png"
    plt.savefig(save_path)
    plt.close() 

    plt.figure(figsize=(12, 8))
    
    for i, (exp_name, history) in enumerate(experiments_history.items()):
        plt.plot(history['val_loss'], label=exp_name, color=colors[i], linewidth=2)
    
    plt.title('Validation Loss Comparison (Lower is Better)', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = f"{save_dir}/comparison_loss.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Graphics saved successfully: {save_dir}/")