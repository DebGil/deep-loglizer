import os
import re
import pickle
import argparse
import pandas as pd
import numpy as np
from utils import decision, json_pretty_dump
from collections import OrderedDict, defaultdict


seed = 42
np.random.seed(seed)

parser = argparse.ArgumentParser()

parser.add_argument("--train_anomaly_ratio", default=0.5, type=float)

params = vars(parser.parse_args())

data_name = f'mylog_{params["train_anomaly_ratio"]}_tar'
data_dir = "../data/processed/mylog_100k"

params = {
    "log_file": "../data/mylog/mylog.log_structured.csv",
    "label_file": "../data/mylog/anomaly_label.csv",
    "test_ratio": 0.8,
    "random_sessions": True,  # shuffle sessions
    "train_anomaly_ratio": params["train_anomaly_ratio"],
}

data_dir = os.path.join(data_dir, data_name)
os.makedirs(data_dir, exist_ok=True)

def extract_sw_ctx_elements(log_line):
    # Regular expression to match the SW_CTX content
    sw_ctx_regex = r'\[SW_CTX:(.*?)\]'
    match = re.search(sw_ctx_regex, log_line)
    if match:
        # Extracting the content inside SW_CTX
        sw_ctx_content = match.group(1)
        # Splitting the content by commas
        sw_ctx_elements = sw_ctx_content.split(',')
        # Selecting the second, third, and fourth elements
        if len(sw_ctx_elements) >= 4:
            return (str(sw_ctx_elements[2]+','+sw_ctx_elements[3]+'.'+ sw_ctx_elements[4]))
    return None, None, None


def preprocess_mylog(
    log_file,
    label_file,
    test_ratio=None,
    train_anomaly_ratio=0.5,
    random_sessions=False,
    **kwargs
):
    """Load HDFS structured log into train and test data

    Arguments
    ---------
        TODO

    Returns
    -------
        TODO
    """
    print("Loading Train Ticket logs from {}.".format(log_file))
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)

    # assign labels
    label_data = pd.read_csv(label_file, engine="c", na_filter=False, memory_map=True)
    label_data["IsError"] = label_data["IsError"].map(
        lambda x: int(x) if isinstance(x, bool) else int(str(x).lower() == "true"))

    average_trace_id_length = struct_log["TraceId"].value_counts().mean()
    min_window_size = int(average_trace_id_length * 0.9)  # adjust the multiplier as needed
    max_window_size = int(average_trace_id_length * 1.1)  # adjust the multiplier as needed

    window_size = np.random.randint(min_window_size, max_window_size)

    label_data_dict = {}
    window_count = 0

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for _, row in enumerate(struct_log.values):

        if window_count not in session_dict:
            session_dict[window_count] = defaultdict(list)
            window_size = np.random.randint(min_window_size, max_window_size)
        session_dict[window_count]["templates"].append(row[column_idx["EventTemplate"]])
        session_dict[window_count]["lineIds"].append(row[column_idx["LineId"]])

        if len(session_dict[window_count]["templates"]) >= window_size:
            window_count += 1

    print('termine de asignar templates')

    total_templates = 0
    total_sessions = len(session_dict)

    # Iterate over each session in session_dict
    for session_id, session_data in session_dict.items():
        # Get the number of templates in the current session
        num_templates = len(session_data["templates"])
        # Add the number of templates to the total
        total_templates += num_templates

    # Calculate the average number of templates per session
    average_templates_per_session = total_templates / total_sessions

    print("Average number of templates per session:", average_templates_per_session)

    for index, row in struct_log.iterrows():
        traceid_spanid_data = label_data.loc[
            (label_data['TraceId'] == row['TraceId']) &
            (label_data['SpanId'] == row['SpanId']) &
            (label_data['Service'] == row['Service'])
            ]
        # Check if any row in the filtered data has isError as True
        if any(traceid_spanid_data['IsError']):
            label_data_dict[ row["LineId"] ] = 1
        else:
            # Otherwise, store False
            label_data_dict[ row["LineId"] ] = 0

    print('termine de asignar labels')

    for window_id, session_data in session_dict.items():
        line_ids = session_data["lineIds"]
        label_assigned = False  # Flag to track if any label has been assigned
        for line_id in line_ids:
            if line_id in label_data_dict:
                label = label_data_dict[line_id]
                if label == 1:
                    session_data["labels"] = label
                    label_assigned = True
                    break  # Move to the next session
                elif label == 0:
                    # Keep checking other line_ids
                    continue
        if not label_assigned:
            # If no label of 1 found, assign 0
            session_data["labels"] = 0

    print('termine de asignar labels a las sesiones')


    session_idx = list(range(len(session_dict)))
    # split data
    if random_sessions:
        print("Using random partition.")
        np.random.shuffle(session_idx)

    session_ids = np.array(list(session_dict.keys()))
    # session_labels = np.array(list(map(lambda x: label_data_dict[x], session_ids)))
    labels_list = []

    # Iterate over each session in session_dict
    for session_data in session_dict.values():
        # Get the 'labels' from the session_data and append to the labels_list
        labels_list.append(session_data['labels'])

    # Convert the labels_list to a NumPy array
    session_labels = np.array(labels_list)

    train_lines = int((1 - test_ratio) * len(session_idx))
    test_lines = int(test_ratio * len(session_idx))

    session_idx_train = session_idx[0:train_lines]
    session_idx_test = session_idx[-test_lines:]

    session_id_train = session_ids[session_idx_train]
    session_id_test = session_ids[session_idx_test]

    print("Total # sessions: {}".format(len(session_ids)))

    print(train_anomaly_ratio , "train anomaly ratio")
    session_train = {
        k: session_dict[k]
        for k in session_id_train
        if (session_dict[k]["labels"] == 0)
        or (session_dict[k]["labels"] == 1 and decision(train_anomaly_ratio))

    }

    session_test = {k: session_dict[k] for k in session_id_test}

    session_labels_train = [v["labels"] for k, v in session_train.items()]
    session_labels_test = [v["labels"] for k, v in session_test.items()]

    train_anomaly = 100 * sum(session_labels_train) / len(session_labels_train)
    test_anomaly = 100 * sum(session_labels_test) / len(session_labels_test)

    print("# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly))
    print("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly))

    with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
        pickle.dump(session_train, fw)
    with open(os.path.join(data_dir, "session_test.pkl"), "wb") as fw:
        pickle.dump(session_test, fw)
    json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))

    print("Saved to {}".format(data_dir))
    return session_train, session_test


if __name__ == "__main__":
    preprocess_mylog(**params)
