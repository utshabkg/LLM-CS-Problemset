"""
Script to preprocess CS-PROB Dataset.xlsx and extract main columns from all tabs.
Columns: Topic, Question, Answer, Difficulty, Source
Output: Combined CSV in data/processed_csprob.csv
"""

import pandas as pd
import os

INPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/CS-PROB Dataset.xlsx'))
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed_csprob.csv'))
MAIN_COLUMNS = ['Topic', 'Question', 'Answer', 'Difficulty', 'Source']

def preprocess_excel(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    dfs = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        # Only keep main columns that exist in this sheet
        cols = [col for col in MAIN_COLUMNS if col in df.columns]
        if cols:
            dfs.append(df[cols])
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")

if __name__ == "__main__":
    preprocess_excel(INPUT_PATH, OUTPUT_PATH)
