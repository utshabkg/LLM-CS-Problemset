"""
Analysis script for LLM evaluation results.
Generates visualizations for:
1. Question Difficulty vs Score for each model
2. Evaluator Model size vs Scores
3. Domain vs Score
4. Human score vs LLM judge scores (first 5 rows from each domain)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../analysis'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_results():
    """Load and merge all evaluation results from the results directory."""
    result_files = [
        'results_llama8B+mistral7B_qwen72B_judge.csv',
        'results_DeepSeek7B_Qwen7B_judge.csv',
        'results_ayaExpanse8B+qwen7B_Qwen72B_judge.csv'
    ]
    
    all_dfs = []
    for file in result_files:
        file_path = os.path.join(RESULTS_DIR, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"Loaded {len(df)} rows from {file}")
                all_dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            print(f"File not found: {file}")
    
    if not all_dfs:
        raise FileNotFoundError("No result files found in results directory")
    
    # Merge all dataframes on common columns
    base_cols = ['Q_ID', 'Domain', 'Difficulty', 'Question', 'Reference_Answer']
    merged_df = all_dfs[0][base_cols].copy()
    
    for df in all_dfs:
        # Get all score and response columns from this df (both _Score and _Score_* patterns)
        score_response_cols = [col for col in df.columns 
                              if '_Score' in col or col.endswith('_Response')]
        for col in score_response_cols:
            if col not in merged_df.columns:
                merged_df = merged_df.merge(df[['Q_ID'] + [col]], on='Q_ID', how='left')
    
    print(f"\nMerged dataset: {len(merged_df)} rows")
    print(f"Total columns: {len(merged_df.columns)}")
    print(f"Columns: {merged_df.columns.tolist()}")
    
    # Standardize Difficulty column (capitalize first letter)
    merged_df['Difficulty'] = merged_df['Difficulty'].str.capitalize()
    
    return merged_df

def extract_model_scores(df):
    """Extract model names and their score columns."""
    # Find all score columns (both _Score and _Score_* patterns)
    score_cols = [col for col in df.columns if '_Score' in col]
    models = {}
    for col in score_cols:
        # Extract model name (everything before _Score)
        model_name = col.split('_Score')[0]
        # Clean up model names for better display
        display_name = model_name.replace('(4bit)', '').replace('_', ' ').strip()
        models[display_name] = col
    return models

def analyze_difficulty_vs_score(df, models):
    """Analyze Question Difficulty vs score of each model and save to CSV."""
    # Filter out NaN values and get unique difficulties
    difficulties = sorted([d for d in df['Difficulty'].unique() if pd.notna(d)], key=str)
    
    results = []
    for model_name, score_col in models.items():
        for diff in difficulties:
            scores = df[df['Difficulty'] == diff][score_col].dropna()
            results.append({
                'Model': model_name,
                'Difficulty': diff,
                'Mean_Score': scores.mean(),
                'Std_Score': scores.std(),
                'Median_Score': scores.median(),
                'Min_Score': scores.min(),
                'Max_Score': scores.max(),
                'Sample_Count': len(scores)
            })
    
    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, 'difficulty_vs_score.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Saved: difficulty_vs_score.csv")
    return results_df

def analyze_model_size_vs_score(df, models):
    """Analyze Evaluator Model size vs Scores and save to CSV."""
    # Define model sizes (in billions of parameters)
    model_sizes = {
        'Meta-Llama-3-8B': 8,
        'Mistral-7B-v0.1': 7,
        'aya-expanse-8b': 8,
        'Qwen2.5-7B-Instruct': 7,
        'aya-expanse-32b': 32,
        'deepseek-llm-7b-chat': 7
    }
    
    results = []
    for model_name, score_col in models.items():
        # Try to match the model name with known sizes
        matched = False
        for size_key in model_sizes.keys():
            if size_key.lower() in model_name.lower() or model_name.lower() in size_key.lower():
                avg_score = df[score_col].mean()
                if not np.isnan(avg_score):
                    results.append({
                        'Model': model_name,
                        'Model_Size_B': model_sizes[size_key],
                        'Mean_Score': avg_score,
                        'Std_Score': df[score_col].std(),
                        'Median_Score': df[score_col].median(),
                        'Min_Score': df[score_col].min(),
                        'Max_Score': df[score_col].max()
                    })
                    matched = True
                    break
        
        if not matched:
            print(f"Warning: Could not find size for model {model_name}")
    
    if len(results) == 0:
        print("No valid model size data found. Skipping model size analysis.")
        return None
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Model_Size_B')
    output_path = os.path.join(OUTPUT_DIR, 'model_size_vs_score.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Saved: model_size_vs_score.csv")
    return results_df

def analyze_domain_vs_score(df, models):
    """Analyze Domain vs Score and save to CSV."""
    # Filter out NaN values and get unique domains
    domains = sorted([d for d in df['Domain'].unique() if pd.notna(d)], key=str)
    
    results = []
    for model_name, score_col in models.items():
        for domain in domains:
            scores = df[df['Domain'] == domain][score_col].dropna()
            results.append({
                'Model': model_name,
                'Domain': domain,
                'Mean_Score': scores.mean(),
                'Std_Score': scores.std(),
                'Median_Score': scores.median(),
                'Min_Score': scores.min(),
                'Max_Score': scores.max(),
                'Sample_Count': len(scores)
            })
    
    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, 'domain_vs_score.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Saved: domain_vs_score.csv")
    return results_df

def create_human_evaluation_template(df):
    """Create a template CSV for human evaluation of first 5 rows from each domain."""
    # Get first 5 rows from each domain
    sample_rows = []
    for domain in df['Domain'].unique():
        domain_df = df[df['Domain'] == domain].head(5)
        sample_rows.append(domain_df)
    
    sample_df = pd.concat(sample_rows, ignore_index=True)
    
    # Create template with necessary columns
    template = sample_df[['Q_ID', 'Domain', 'Difficulty', 'Question', 'Reference_Answer']].copy()
    
    # Add model response columns
    response_cols = [col for col in df.columns if col.endswith('_Response')]
    for col in response_cols:
        if col in sample_df.columns:
            template[col] = sample_df[col]
    
    # Add model score columns
    score_cols = [col for col in df.columns if col.endswith('_Score')]
    for col in score_cols:
        if col in sample_df.columns:
            template[col] = sample_df[col]
    
    # Add human score columns (empty for manual filling)
    for col in score_cols:
        human_col = col.replace('_Score', '_Human_Score')
        template[human_col] = np.nan
    
    # Save template
    template_path = os.path.join(OUTPUT_DIR, 'human_evaluation_template.csv')
    template.to_csv(template_path, index=False)
    print(f"\nCreated human evaluation template: human_evaluation_template.csv")
    print(f"Please fill in the Human_Score columns (0.0 to 1.0) and save as 'human_evaluation_completed.csv'")
    print(f"Total rows to evaluate: {len(template)}")
    
    return template_path

def plot_human_vs_llm_scores(df, models):
    """Plot 4: Human score vs LLM judge scores (requires human_evaluation_completed.csv)."""
    completed_path = os.path.join(OUTPUT_DIR, 'human_evaluation_completed.csv')
    
    if not os.path.exists(completed_path):
        print("\n" + "="*80)
        print("Human evaluation file not found!")
        print(f"Please complete the human evaluation in: {completed_path}")
        print("="*80)
        return
    
    # Load human-evaluated data
    human_df = pd.read_csv(completed_path)
    
    # Extract human score columns
    human_score_cols = [col for col in human_df.columns if col.endswith('_Human_Score')]
    
    if not human_score_cols:
        print("No human score columns found in the completed file!")
        return
    
    # Create subplots for each model
    n_models = len(models)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Human Scores vs LLM Judge Scores', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, (model_name, score_col) in enumerate(models.items()):
        if idx >= len(axes):
            break
        
        human_col = score_col.replace('_Score', '_Human_Score')
        
        if human_col not in human_df.columns:
            continue
        
        ax = axes[idx]
        
        # Get valid pairs (non-NaN)
        valid_mask = human_df[human_col].notna() & human_df[score_col].notna()
        human_scores = human_df[valid_mask][human_col]
        llm_scores = human_df[valid_mask][score_col]
        
        if len(human_scores) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_name)
            continue
        
        # Scatter plot
        ax.scatter(human_scores, llm_scores, alpha=0.6, s=100)
        
        # Add diagonal line (perfect agreement)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Agreement')
        
        # Calculate correlation
        if len(human_scores) > 1:
            correlation = np.corrcoef(human_scores, llm_scores)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Human Score', fontweight='bold')
        ax.set_ylabel('LLM Judge Score', fontweight='bold')
        ax.set_title(f'{model_name}', fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_human_vs_llm_scores.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: 4_human_vs_llm_scores.png")
    plt.close()

def generate_summary_statistics(df, models):
    """Generate summary statistics table."""
    summary_data = []
    
    for model_name, score_col in models.items():
        scores = df[score_col].dropna()
        summary_data.append({
            'Model': model_name,
            'Mean Score': scores.mean(),
            'Std Dev': scores.std(),
            'Median': scores.median(),
            'Min': scores.min(),
            'Max': scores.max(),
            'Count': len(scores)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, 'summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary statistics: summary_statistics.csv")
    print(summary_df.to_string(index=False))

def main():
    print("="*80)
    print("LLM Evaluation Results Analysis (CSV Output Only)")
    print("="*80)
    
    # Load data
    df = load_results()
    
    # Extract models
    models = extract_model_scores(df)
    print(f"\nFound {len(models)} models: {list(models.keys())}")
    
    # Generate CSV analyses
    print("\nGenerating CSV analyses...")
    analyze_difficulty_vs_score(df, models)
    analyze_model_size_vs_score(df, models)
    analyze_domain_vs_score(df, models)
    
    # Generate summary statistics
    generate_summary_statistics(df, models)
    
    # Create human evaluation template
    print("\n" + "="*80)
    create_human_evaluation_template(df)
    
    print("\n" + "="*80)
    print("Analysis complete! Check the 'analysis' folder for CSV outputs.")
    print("="*80)

if __name__ == "__main__":
    main()
