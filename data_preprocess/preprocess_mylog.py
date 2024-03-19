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

parser.add_argument("--train_anomaly_ratio", default=0.1, type=float)

params = vars(parser.parse_args())

data_name = f'mylog_{params["train_anomaly_ratio"]}_tar'
data_dir = "../data/processed/mylog_100k"

params = {
    "log_file": "../data/mylog/mylog.log_structured.csv",
    "label_file": "../data/mylog/anomaly_label.csv",
    "test_ratio": 0.9,
    "random_sessions": False,  # shuffle sessions
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


    distinct_combinations = struct_log.drop_duplicates(subset=["TraceId", "SpanId","Service"])
    print("Number of distinct combinations of TraceId, SpanID and Service in logs:", distinct_combinations.shape[0])


    label_data_dict = {}

    unique_combinations = label_data.drop_duplicates(subset=["TraceId", "SpanId","Service"])
    print("Number of distinct combinations of TraceId, SpanID and Service in labels:", unique_combinations.shape[0])

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for _, row in enumerate(struct_log.values):
        # trace_id = row[column_idx["TraceId"]]
        # span_id = row[column_idx["SpanId"]]
        # service = row[column_idx["Service"]]
        #
        # id_tuple = (trace_id, span_id, service)

        id_list= row[column_idx["TraceId"]]+','+row[column_idx["SpanId"]]+','+row[column_idx["Service"]]
        if id_list not in session_dict:
            session_dict[id_list] = defaultdict(list)
        session_dict[id_list]["templates"].append(row[column_idx["EventTemplate"]])

    print('termine de asignar templates')

    # Iterate over unique combinations
    for index, row in distinct_combinations.iterrows():
        traceid_spanid_data = label_data.loc[
            (label_data['TraceId'] == row['TraceId']) &
            (label_data['SpanId'] == row['SpanId']) &
            (label_data['Service'] == row['Service'])
            ]
        # Check if any row in the filtered data has isError as True
        if any(traceid_spanid_data['IsError']):
            # label_data_dict[(row['TraceId'], row['SpanId'], row['Service'])] = 1
            label_data_dict[row["TraceId"] + ',' + row["SpanId"] + ',' + row["Service"]] = 1
        else:
            # Otherwise, store False
            # label_data_dict[(row['TraceId'], row['SpanId'], row['Service'])] = 0
            label_data_dict[row["TraceId"] + ',' + row["SpanId"] + ',' + row["Service"]] = 0

    print('termine de asignar labels')


    # label_data_dict = dict(zip(label_data["TraceId"]+','+label_data["SpanId"]+','+label_data["Service"], label_data["IsError"]))


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
