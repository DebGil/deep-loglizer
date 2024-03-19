import os
import re
import pickle
import argparse
import random
import pandas as pd
import numpy as np
from utils import decision, json_pretty_dump
from collections import OrderedDict, defaultdict


seed = 42
np.random.seed(seed)

parser = argparse.ArgumentParser()

parser.add_argument("--train_anomaly_ratio", default=0.1, type=float)

params = vars(parser.parse_args())

data_name = f'mylog_sampled_{params["train_anomaly_ratio"]}_tar'
data_dir = "../data/processed/mylog_100k"

params = {
    "log_file": "../data/mylog/mylog.log_structured.csv",
    "label_file": "../data/mylog/anomaly_label.csv",
    "test_ratio": 0.1,
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


    label_data_dict = {}

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for _, row in enumerate(struct_log.values):

        id_list= row[column_idx["TraceId"]]
        if id_list not in session_dict:
            session_dict[id_list] = defaultdict(list)
        session_dict[id_list]["templates"].append(row[column_idx["EventTemplate"]])

    print('termine de asignar templates')

    total_templates = 0
    total_sessions = len(session_dict)

    total_templates = sum(len(session_data["templates"]) for session_data in session_dict.values())

    # Calculate the number of templates to drop (10% of total templates)
    templates_to_drop = int(0.1 * total_templates)

    # Create a list of all template indices
    all_template_indices = [(session_id, idx) for session_id, session_data in session_dict.items()
                            for idx, _ in enumerate(session_data["templates"])]

    # Randomly select templates to drop
    templates_indices_to_drop = random.sample(all_template_indices, templates_to_drop)

    # Remove the selected templates from session_dict
    for session_id, index_to_drop in sorted(templates_indices_to_drop, reverse=True):
        session_dict[session_id]["templates"].pop(index_to_drop)

    # Recalculate the total number of templates after dropping
    total_templates_after_drop = sum(len(session_data["templates"]) for session_data in session_dict.values())

    print("Total number of templates before dropping:", total_templates)
    print("Total number of templates after dropping:", total_templates_after_drop)

    # Iterate over unique combinations
    for index, row in struct_log.iterrows():
        print(index
              )
        traceid_data = label_data.loc[(label_data['TraceId'] == row['TraceId'])]
        # Check if any row in the filtered data has isError as True
        if any(traceid_data['IsError']):
            label_data_dict[ row["TraceId"] ] = 1
        else:
            # Otherwise, store False
            label_data_dict[ row["TraceId"] ] = 0

    print('termine de asignar labels')




    for k in list(session_dict.keys()):
        try:
            session_dict[k]["label"] = label_data_dict[k]
        except KeyError:
            del session_dict[k]
            continue
    print('termine de asignar labels a las sesiones')


    session_idx = list(range(len(session_dict)))
    # split data
    if random_sessions:
        print("Using random partition.")
        np.random.shuffle(session_idx)

    session_ids = np.array(list(session_dict.keys()))
    session_labels = np.array(list(map(lambda x: label_data_dict[x], session_ids)))

    train_lines = int((1 - test_ratio) * len(session_idx))
    test_lines = int(test_ratio * len(session_idx))

    session_idx_train = session_idx[0:train_lines]
    session_idx_test = session_idx[-test_lines:]

    session_id_train = session_ids[session_idx_train]
    session_id_test = session_ids[session_idx_test]
    session_labels_train = session_labels[session_idx_train]
    session_labels_test = session_labels[session_idx_test]

    print("Total # sessions: {}".format(len(session_ids)))

    session_train = {
        k: session_dict[k]
        for k in session_id_train
        if (session_dict[k]["label"] == 0)
        or (session_dict[k]["label"] == 1 and decision(train_anomaly_ratio))

    }

    session_test = {k: session_dict[k] for k in session_id_test}

    session_labels_train = [v["label"] for k, v in session_train.items()]
    session_labels_test = [v["label"] for k, v in session_test.items()]

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
