
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

def generate_charts():
    json_path = '/home/iclab/slm_iclab-1/intern_summer/metrics_outputs/dataset_eval_results.json'
    output_dir = '/home/iclab/slm_iclab-1/intern_summer/metrics_outputs/charts'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    datasets = data['metadata']['datasets']
    models = data['models']
    
    # helper to find pairs
    base_models = []
    release_models = []
    
    for model_name, model_data in models.items():
        if model_data.get('group') == 'base':
            base_models.append((model_name, model_data))
        elif model_data.get('group') == 'release':
            release_models.append((model_name, model_data))
            
    # sort for consistent order
    base_models.sort(key=lambda x: x[0])
    
    # Pre-calculate pairs to handle them consistently across datasets
    model_pairs = []
    for base_name, base_data in base_models:
        matching_release = None
        for rel_name, rel_data in release_models:
            if rel_name.startswith(base_name):
                matching_release = (rel_name, rel_data)
                break
        
        if matching_release:
            model_pairs.append({
                'base_name': base_name,
                'base_data': base_data,
                'release_name': matching_release[0],
                'release_data': matching_release[1]
            })
        else:
            print(f"Warning: No matching release model found for {base_name}")

    # Now generate a chart and CSV for each dataset
    for dataset_key, dataset_info in datasets.items():
        dataset_display_name = dataset_info.get('display', dataset_key)
        
        labels = []
        base_scores = []
        release_scores = []
        
        for pair in model_pairs:
            base_name = pair['base_name']
            
            # Get metrics for this dataset
            base_metric = pair['base_data']['metrics'].get(dataset_key, {})
            release_metric = pair['release_data']['metrics'].get(dataset_key, {})
            
            # We use accuracy as the "score"
            b_score = base_metric.get('accuracy', 0)
            r_score = release_metric.get('accuracy', 0)
            
            # Shorten name for display
            display_name = base_name
            if "HuggingFaceTB_" in display_name:
                display_name = display_name.replace("HuggingFaceTB_", "")
            if "ibm-granite_" in display_name:
                 display_name = display_name.replace("ibm-granite_", "")
            if "TinyLlama_" in display_name:
                display_name = display_name.replace("TinyLlama_", "")
                
            labels.append(display_name)
            base_scores.append(b_score)
            release_scores.append(r_score)
            
        if not labels:
            continue
            
        # Plotting
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, base_scores, width, label='Base Model', color='#4e79a7') # blue-ish
        rects2 = ax.bar(x + width/2, release_scores, width, label='Release Model', color='#f28e2b') # orange-ish
        
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Model Comparison on {dataset_display_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels
        ax.bar_label(rects1, padding=3, fmt='%.2f')
        ax.bar_label(rects2, padding=3, fmt='%.2f')
        
        plt.tight_layout()
        
        filename = f"{dataset_key}_comparison.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Generated chart: {save_path}")
        
        # CSV Generation
        csv_filename = f"{dataset_key}_comparison.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model', 'Base Accuracy', 'Release Accuracy'])
            for i in range(len(labels)):
                writer.writerow([labels[i], base_scores[i], release_scores[i]])
        print(f"Generated CSV: {csv_path}")

if __name__ == "__main__":
    generate_charts()
