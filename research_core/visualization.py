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

    
    selected_experiments = [

        "EXP_01_Shallow_BCE",         
        "EXP_06_Deep_BCE",            
        "EXP_11_ResNet_BCE",          

        "EXP_14_ResNet_Poly",        
        "EXP_13_ResNet_Focal",        
        "EXP_08_Deep_Focal",        

        "EXP_19_Weighted_W100",      
        "EXP_16_ResNet_Focal_G05",    
        "EXP_02_Shallow_WeightedBCE"  
    ]

    style_map = {

        "EXP_01_Shallow_BCE":       {"color": "lightblue", "style": "--", "label": "Shallow BCE"},
        "EXP_06_Deep_BCE":          {"color": "dodgerblue", "style": "-.", "label": "Deep BCE"},
        "EXP_11_ResNet_BCE":        {"color": "darkblue",   "style": "-",  "label": "ResNet BCE"},

        "EXP_14_ResNet_Poly":       {"color": "red",        "style": "-",  "label": "(PolyLoss)"}, 
        "EXP_13_ResNet_Focal":      {"color": "orange",     "style": "-",  "label": "ResNet Focal"},
        "EXP_08_Deep_Focal":        {"color": "salmon",     "style": "-.", "label": "Deep Focal"},

        "EXP_19_Weighted_W100":     {"color": "cyan",       "style": ":",  "label": "High Weight (W=100)"},
        "EXP_16_ResNet_Focal_G05":  {"color": "black",      "style": ":",  "label": "Unstable (gamma=0.5)"},
        "EXP_02_Shallow_WeightedBCE":{"color": "gray",      "style": "--", "label": "Shallow Weighted"}
    }


    metrics_config = [
        ('val_prauc', 'Validation PR-AUC Comparison', 'comparison_prauc.png', 'PR-AUC Score'),
        ('val_loss', 'Validation Loss Comparison', 'comparison_loss.png', 'Loss'),
        ('val_f1', 'Validation F1-Score Comparison', 'comparison_f1.png', 'F1-Score'),
        ('val_recall', 'Validation Recall Comparison', 'comparison_recall.png', 'Recall (Sensitivity)'),
        ('val_precision', 'Validation Precision Comparison', 'comparison_precision.png', 'Precision')
    ]

    print(f"\nGenerating graphics: {save_dir} ...")

    for metric_key, title, filename, ylabel in metrics_config:
        plt.figure(figsize=(12, 7)) 
        
        for exp_name in selected_experiments:
         
            found = False
            for actual_name in experiments_history.keys():
                if actual_name == exp_name:
                    found = True
                    break
            
            if found and metric_key in experiments_history[exp_name]:
                history = experiments_history[exp_name][metric_key]
                style = style_map.get(exp_name, {"color": "gray", "style": "-", "label": exp_name})
                
                lw = 2.5 if "ResNet" in exp_name or "WINNER" in style["label"] else 1.5
                
                plt.plot(history, 
                         label=style["label"], 
                         color=style["color"], 
                         linestyle=style["style"], 
                         linewidth=lw)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10)
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"Graphics saved successfully: {save_dir}/")

