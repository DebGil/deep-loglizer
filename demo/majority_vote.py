import sys
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def load_pickle(filename):
    """Load data from a pickle file into an array of integers."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data.astype(int)

def majority_vote(file1, file2):
    """Perform majority vote on two arrays of integers."""
    array1 = load_pickle(file1)
    array2 = load_pickle(file2)

    # Perform majority vote
    majority_result = []
    for value1, value2 in zip(array1, array2):
        majority_result.append(max(value1, value2))  # Choose the maximum value

    return majority_result

def calculate_metrics(majority_result, file3):
    array3 = load_pickle(file3)
    eval_results = {
        "f1": f1_score(array3, majority_result),
        "rc": recall_score(array3, majority_result),
        "pc": precision_score(array3, majority_result),
        "acc": accuracy_score(array3, majority_result),
    }
    return eval_results

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python majority_vote.py <file1> <file2> <file3>")
        sys.exit(1)

    # Get the filenames of the pickle files from command-line arguments
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]

    # Perform majority vote on the pickle files
    majority_result = majority_vote(file1, file2)

    # Print the majority vote result
    print("Majority vote result:")
    print(majority_result)

    eval_results = calculate_metrics(majority_result, file3)
    print({k: f"{v:.3f}" for k, v in eval_results.items()})

