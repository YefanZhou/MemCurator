import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_analysis_results(steps, avg_scores, update_percentages, other_op_percentages, output_dir):
    """
    Plots the analysis results for average scores and update percentages over steps.

    Args:
        steps (list): List of step numbers.
        avg_scores (list): List of average scores corresponding to the steps.
        update_percentages (list): List of update percentages corresponding to the steps.
        other_op_percentages (list): List of percentages for other operations.
        output_dir (str): The directory to save the plot image.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting average scores on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Average Score (Update Op)', color=color)
    ax1.plot(steps, avg_scores, color=color, marker='o', linestyle='-', label='Average Score (Update)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    # Creating a secondary y-axis for the update percentage
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Operation Percentage (%)')
    ax2.plot(steps, update_percentages, color=color, marker='x', linestyle='--', label='Update Op %')
    ax2.tick_params(axis='y')

    # Add plot for other operations
    color_other = 'tab:green'
    ax2.plot(steps, other_op_percentages, color=color_other, marker='s', linestyle=':', label='Other Ops %')

    plt.title('Average Score and Operation Percentage per Step')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    # Save the plot
    plot_filename = os.path.join(output_dir, 'skill_update_analysis.png')
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")
    plt.close(fig)
def analyze_skill_update_scores(directory):
    """
    Calculates the average score of JSON objects in .jsonl files that contain
    a specific skill_update function call in their output, grouped by step
    from the filename.

    Args:
        directory (str): The path to the directory containing .jsonl files.
    """
    scores_by_step = {}
    update_counts_by_step = {}
    other_op_counts_by_step = {}
    total_counts_by_step = {}
    update_target_string = "✿FUNCTION✿: memory_update\n✿ARGS✿:"
    other_op_strings = [
        "✿FUNCTION✿: new_memory_insert\n✿ARGS✿:",
        "✿FUNCTION✿: memory_delete\n✿ARGS✿:"
    ]

    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    print(f"Analyzing files in '{directory}'...")

    # Get all jsonl files and sort them numerically by filename (step number)
    try:
        files_to_process = sorted(
            [f for f in os.listdir(directory) if f.endswith('.jsonl')],
            key=lambda f: int(os.path.splitext(f)[0])
        )[:15]
    except ValueError:
        print("Error: Could not sort files numerically. Ensure filenames are integers (e.g., '1.jsonl', '10.jsonl').")
        # Fallback to alphabetical sort if numeric sort fails
        files_to_process = sorted([f for f in os.listdir(directory) if f.endswith('.jsonl')])

    for filename in files_to_process:
        try:
            step_number = int(os.path.splitext(filename)[0])
        except ValueError:
            print(f"Warning: Could not parse step number from filename '{filename}'. Skipping.")
            continue

        filepath = os.path.join(directory, filename)
        step_scores = []
        update_count = 0
        other_op_count = 0
        total_count = 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    total_count += 1
                    try:
                        data = json.loads(line)
                        output = data.get('output', '')
                        if update_target_string in output:
                            update_count += 1
                            if 'score' in data:
                                step_scores.append(data['score'])

                        for op_str in other_op_strings:
                            if op_str in output:
                                other_op_count += 1
                                break
                    except json.JSONDecodeError:
                        # Silently ignore JSON decoding errors for cleaner output
                        pass
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")

        if step_scores:
            scores_by_step[step_number] = step_scores
        if total_count > 0:
            other_op_counts_by_step[step_number] = other_op_count
            update_counts_by_step[step_number] = update_count
            total_counts_by_step[step_number] = total_count

    if scores_by_step:
        print("\n--- Analysis Complete ---")
        
        steps = sorted(scores_by_step.keys())
        all_scores_combined = []
        avg_scores = []
        update_percentages = []
        other_op_percentages = []
        
        # Iterate through sorted steps to print results in order
        for step in steps:
            scores = scores_by_step[step]
            average_score = np.mean(scores)
            all_scores_combined.extend(scores)
            avg_scores.append(average_score)

            update_count = update_counts_by_step.get(step, 0)
            total_count = total_counts_by_step.get(step, 1)
            percentage = (update_count / total_count) * 100 if total_count > 0 else 0
            update_percentages.append(percentage)

            other_op_count = other_op_counts_by_step.get(step, 0)
            other_percentage = (other_op_count / total_count) * 100 if total_count > 0 else 0
            other_op_percentages.append(other_percentage)

            print(f"Step {step}: Found {len(scores)} update entries out of {total_count}. "
                  f"Avg Score: {average_score:.4f}, Update %: {percentage:.2f}%, Other Ops %: {other_percentage:.2f}%")

        if all_scores_combined:
            overall_average = np.mean(all_scores_combined)
            print("\n--- Overall ---")
            print(f"Found a total of {len(all_scores_combined)} entries across all steps.")
            print(f"Overall average score: {overall_average:.4f}")

        # Plotting the results
        plot_analysis_results(steps, avg_scores, update_percentages, other_op_percentages, directory)

    else:
        print("\n--- Analysis Complete ---")
        print("No entries with 'skill_update' function call were found in any file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze skill_update scores from .jsonl files.")
    parser.add_argument(
        'directory',
        type=str,
        help="The directory to search for .jsonl files.",
        nargs='?',
        default='./math/qwen2.5-7b-grouped+content0.5+compression0.1'
    )
    args = parser.parse_args()

    analyze_skill_update_scores(args.directory)
