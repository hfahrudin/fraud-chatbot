import pandas as pd
import requests
import os
from dotenv import load_dotenv
import numpy as np
import sqlparse
from rouge_score import rouge_scorer
import argparse
import time
import re
# Load environment variables
load_dotenv()


def mask_word_after_as(text_string):
    """
    1. Finds the first word after 'AS' or 'as'.
    2. Masks ALL occurrences of that specific word with '***' throughout the string.
    
    Args:
        text_string: The input string to be processed.

    Returns:
        The modified string with the target word masked, or the original string 
        if no 'AS' pattern is found.
    """
    match = re.search(r'as\s(\w+)', text_string, flags=re.IGNORECASE)
    
    if match:
        # Extract the word captured in Group 1 (the word after 'as ')
        target_word = match.group(1)
        
        pattern_to_mask = r'\b' + re.escape(target_word) + r'\b'
        masked_string = re.sub(pattern_to_mask, '***', text_string, flags=re.IGNORECASE)
        
        return masked_string
    
    else:
        # No 'AS <word>' pattern was found
        return text_string

def run_evaluation(csv_file: str):
    print("--- Starting Evaluation ---")
    
    # Load the dataset
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Dataset file '{csv_file}' not found.")
        return

    # SQL evaluation variables
    sql_correct_predictions = 0
    sql_total = 0
    
    # RAG evaluation variables
    rag_rouge_scores = []
    rag_total = 0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for index, row in df.iterrows():
        query = row['query']
        print(f"Processing query: {query}")

        try:
            response = requests.post("http://localhost:8000/eval", json=[{"role": "user", "content": query}])
            response_json = response.json()
            
            # --- SQL Evaluation Logic ---
            if pd.notna(row['expected_sql_query']):
                sql_total += 1
                expected_sql = row['expected_sql_query']
                if response_json.get('tool_calls'):
                    generated_sql = response_json['tool_calls'][0]['query']
                    print(f"  Expected SQL: {expected_sql}")
                    print(f"  Generated SQL: {generated_sql}")

                    normalized_expected_sql = sqlparse.format(expected_sql, keyword_case='lower', identifier_case='lower', strip_comments=True, reindent=True).rstrip(';')
                    normalized_generated_sql = sqlparse.format(generated_sql, keyword_case='lower', identifier_case='lower', strip_comments=True, reindent=True).rstrip(';')
                    
                    normalized_expected_sql = mask_word_after_as(normalized_expected_sql)
                    normalized_generated_sql = mask_word_after_as(normalized_generated_sql)

                    if normalized_generated_sql == normalized_expected_sql:
                        sql_correct_predictions += 1
                        print("  SQL Result: Correct")
                    else:
                        print("  SQL Result: Incorrect")
                else:
                    print("  SQL Result: No SQL query generated")

            # --- RAG Evaluation Logic ---
            if pd.notna(row['expected_rag_response']):
                rag_total += 1
                answer = response_json.get('final_answer', '')
                ground_truth = row['expected_rag_response']
                
                print(f"  RAG Answer: {answer}")
                
                scores = scorer.score(ground_truth, answer)
                rag_rouge_scores.append(scores['rougeL'].fmeasure)
                print(f"  RAG ROUGE-L F1: {scores['rougeL'].fmeasure:.4f}")


        except requests.exceptions.RequestException as e:
            print(f"Error calling /eval endpoint for query '{query}': {e}")

        print("-" * 20)

    # --- Final SQL Evaluation Summary ---
    if sql_total > 0:
        accuracy = (sql_correct_predictions / sql_total) * 100
        print(f"\n--- SQL Evaluation Summary ---")
        print(f"Accuracy: {accuracy:.2f}% ({sql_correct_predictions}/{sql_total})")
    else:
        print("\nNo SQL queries were evaluated.")

    # --- Final RAG Evaluation Summary ---
    if rag_total > 0:
        average_rouge_score = np.mean(rag_rouge_scores)
        print(f"\n--- RAG Evaluation Summary ---")
        print(f"Average ROUGE-L F1 Score: {average_rouge_score:.4f}")
    else:
        print("\nNo RAG questions were evaluated.")

    time.sleep(2)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SQL and RAG evaluations.")
    parser.add_argument("--csv_file", type=str, default="eval/evaluation_dataset.csv",
                        help="Path to the CSV file containing evaluation data.")
    args = parser.parse_args()
    run_evaluation(args.csv_file)