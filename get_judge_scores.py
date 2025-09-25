import pandas as pd
import json
import os
import sys
import requests
import time
from typing import Dict, Any, List, Tuple
import tiktoken
import math
import signal
import argparse
import numpy as np
# from utils.summ_eval import create_summeval_prompt, format_result
from utils.secure_gpt import completion_with_backoff, extract_score, format_evaluation, completion_with_backoff_anthropic
from utils.datasets.summ_eval import SummevalDataset
from utils.datasets.hanna import HannaDataset
from utils.datasets.mslr import MSLRDataset
from utils.datasets.medval import MedValDataset

def timeout_handler(signum, frame):
    raise TimeoutError("Evaluation timed out")

def evaluate_text(prompt_text: str, judge_model="openai") -> Dict[str, Any]:
    if judge_model == "openai":
        response = completion_with_backoff(
        messages=[
            {"role": "system", "content": f"You are an AI assistant tasked with evaluating text."},
            {"role": "user", "content": prompt_text}
        ],
        max_tokens=1000,
        temperature=0.2,
    )
        try:
            evaluation = json.loads(response['choices'][0]['message']['content'])
            return evaluation
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing LLM response: {e}")
            print("Raw response:")
            print(response)
            return None
    
    elif judge_model == "anthropic":
        response = completion_with_backoff_anthropic(messages={"model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0", "prompt_text": prompt_text})
        try:
            evaluation= response['content'][0]['text']
            return evaluation
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing LLM response: {e}")
            print("Raw response:")
            print(response)
            return None
    else:
        raise ValueError(f"Invalid judge model: {judge_model}")


def calculate_score_differences(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate average absolute and raw differences between human and model scores."""
    total_diff = 0  # Raw difference (model - human)
    total_abs_diff = 0  # Absolute difference |model - human|
    count = 0
    for result in results:
        evaluation = result.get('evaluation', {})
        try:
            if type(evaluation) == str:
                evaluation = json.loads(evaluation)
            model_score = evaluation['evaluation']['score']
        except:
            model_score = evaluation['score']
        human_score = result.get('original_score')
        
        if model_score is not None and human_score is not None:
            diff = model_score - human_score
            total_diff += diff
            total_abs_diff += abs(diff)
            count += 1

    if count == 0:
        return {
            "average_difference": 0,
            "average_absolute_difference": 0,
            "count": 0
        }

    return {
        "average_difference": total_diff / count,  # Positive means model scores higher
        "average_absolute_difference": total_abs_diff / count,
        "count": count
    }

def judge_pipeline(results, dataset_obj, processed_ids, output_file, dimension=None, judge_model="openai"):
    """
    This function is used to judge the text using the LLM.
    It will evaluate the text and save the results to the output file.
    It will also calculate the score differences and save them to the output file.
    """
    print(f"Processing {len(dataset_obj.get_data_and_prompts())} texts")
    data_and_prompts = dataset_obj.get_data_and_prompts()
    for idx, data_row in data_and_prompts.iterrows():
        if data_row["text_id"] in processed_ids:
            print(f"Text {data_row['text_id']} already processed. Skipping.")
            continue
        try:
            print(f"Evaluating text {data_row['text_id']}")
            signal.alarm(300)  # 5-minute timeout

            evaluation = evaluate_text(data_row['prompt_text'], judge_model)
            if evaluation:
                result_dict = dataset_obj.format_result(evaluation, data_row)
                try:
                    model_score = evaluation['evaluation']['score']
                except:
                    evaluation = json.loads(evaluation)
                    model_score = evaluation['evaluation']['score']
                if model_score is not None:
                    result_dict["score_difference"] = model_score - float(data_row['original_score'])

                results.append(result_dict)

                # Save results after each evaluation
                score_diffs = calculate_score_differences(results)
                with open(output_file, "w") as f:
                    json.dump({
                        "dimension": dimension,
                        "score_differences": score_diffs,
                        "detailed_results": results
                    }, f, indent=2)

            signal.alarm(0)  # Reset the alarm

        except TimeoutError:
            print(f"Evaluation timed out for text {idx}")
        except Exception as e:
            print(f"Error processing text {idx}: {str(e)}")
    return results

def print_results(results, output_file):
    """
    This function is used to print the results.
    """
    # Final score differences calculation
    score_diffs = calculate_score_differences(results)
    print(f"Evaluation complete. Results saved to {output_file}")
    print(f"Average difference (model - human): {score_diffs['average_difference']:.2f}")
    print(f"Average absolute difference: {score_diffs['average_absolute_difference']:.2f}")
    print(f"Number of evaluations: {score_diffs['count']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', required=True, type=str, help='Output JSON file for results')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of samples to evaluate (default: all)')
    parser.add_argument('--dataset', type=str, default='summeval', help='Dataset to evaluate (default: summeval)')
    parser.add_argument('--dimension', type=str, default=None, help='Specific dimension to evaluate (default: evaluate all dimensions)')
    args = parser.parse_args()

    # Create initial dataset object to get dimensions
    if args.dataset == 'summeval':
        dataset_class = SummevalDataset
    elif args.dataset == 'hanna':
        dataset_class = HannaDataset
    elif args.dataset == 'mslr':
        dataset_class = MSLRDataset
    elif args.dataset == 'medval':
        dataset_class = MedValDataset
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # Get dimensions to evaluate
    dimensions = [args.dimension] if args.dimension else dataset_class(None, None).get_available_dimensions()

    for dimension in dimensions:
        # Create dataset object for this dimension
        dataset_obj = dataset_class(dimension, args.sample_size)
        
        # Modify output file path to include dimension
        dimension_output_file = args.output_file.replace('.json', f'_{dimension}.json')

        results = []
        processed_ids = set()
        if os.path.exists(dimension_output_file):
            with open(dimension_output_file, 'r') as f:
                existing_results = json.load(f)
            results = existing_results.get('detailed_results', [])
            processed_ids = set(result.get('text_id') for result in results)
            breakpoint()
        
        signal.signal(signal.SIGALRM, timeout_handler)
        results = judge_pipeline(results, dataset_obj, processed_ids, dimension_output_file, dimension, judge_model="anthropic")
        print_results(results, dimension_output_file)

if __name__ == "__main__":
    main()
