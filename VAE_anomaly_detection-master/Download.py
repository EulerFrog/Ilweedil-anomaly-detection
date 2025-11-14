import requests
from requests.auth import HTTPBasicAuth
import json
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data(username, password):
    # Suppress HTTPS warnings (because verify=False)
    urllib3.disable_warnings(InsecureRequestWarning)
    
    # OpenSearch URL
    url = "https://140.160.1.66:9200/_search"
    #url = "https://140.160.1.66:9200/_cat/indices?v"
    #url = "https://140.160.1.66:9200/_cat/indices?format=json"
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"event.module": "suricata"}}
                ],
            }
        },
        "size": 10000
    }
    
    # Send the request
    response = requests.get(
        url,
        headers={"Content-Type": "application/json"},
        auth=HTTPBasicAuth(username, password),
        data=json.dumps(query),
        verify=False  # ignore SSL certificate
    )

    return json.loads(response.text)

def populate(raw_data, data):
  
    columns = data.keys()

    for hit in raw_data:
        
        flow_dict = hit["_source"]

        for col in columns:

            if col == "direction":
                # 0 if to_server, 1 if to_client
                data[col].append(0.0 if flow_dict["suricata"][col] == "to_server" else 1.0)
            elif col == "duration":
                duration = (int(flow_dict["event"]["end"]) - int(flow_dict["event"]["start"])) / 1000 # convert from ms to sec
                data[col].append(duration)
            elif col == "label":
                data[col].append(0.0 if not flow_dict["suricata"]["alert"] else 1.0)
            elif col == "protocol":
                data[col].append(flow_dict[col][0])
            else:
                data[col].append(flow_dict["suricata"]["flow"][col])

    return data

def remap(data):
    common_ports = {
        "80": "HTTP",
        "8080": "HTTP",
        "443": "HTTPS",
        "22": "SSH",
        "922": "SSH",
        "53": "DNS",
        "67": "DHCP",
        "68": "DHCP",
        "25": "SMTP",
        "161": "SNMP",
        "162": "SNMP",
        "3389": "RDP",
        "3306": "SQL",
        "20": "FTP",
        "21": "FTP",
    }

    for idx, ports in enumerate(zip(data["src_port"], data["dest_port"])):
        src_port, dest_port = ports
        if str(src_port) in common_ports.keys():
            data["src_port"][idx] = common_ports[str(src_port)]
        elif src_port < 1024:
            data["src_port"][idx] = "public"
        elif src_port < 49152:
            data["src_port"][idx] = "private"
        elif src_port > 49151:
            data["src_port"][idx] = "dynamic"

        if str(dest_port) in common_ports.keys():
            data["dest_port"][idx] = common_ports[str(dest_port)]
        elif dest_port < 1024:
            data["dest_port"][idx] = "public"
        elif dest_port < 49152:
            data["dest_port"][idx] = "private"
        elif dest_port > 49151:
            data["dest_port"][idx] = "dynamic"

    return data

def one_hot(df):
    categories = {
        "src_port": ["HTTP", "HTTPS", "SSH", "DNS", "DHCP", "SMTP", "SNMP", "RDP", "SQL", "FTP", "public", "private", "dynamic"],
        "dest_port": ["HTTP", "HTTPS", "SSH", "DNS", "DHCP", "SMTP", "SNMP", "RDP", "SQL", "FTP", "public", "private", "dynamic"],
        "protocol": ["tcp", "udp", "icmp", "ipv6-icmp", "gre", "esp", "other"],
    }

    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Create encoder with fixed categories and ignore unseen
    encoder = OneHotEncoder(categories=[categories[col] for col in categorical_columns],
                            handle_unknown='ignore', sparse_output=False)

    # fit and transform list values to columns in the encoder
    encoded = encoder.fit_transform(df[categorical_columns])
    # create one hot encoded df (only categorical columns)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns))
    # concat with original df
    df_encoded = pd.concat([encoded_df, df], axis=1)
    # delete columns that we don't want anymore (duplicates)
    df_encoded = df_encoded.drop(categorical_columns, axis=1)
    #print(f"Encoded data : \n{df_encoded}")

    return df_encoded

def min_max(df, col):
    col_min = df[col].min()
    col_max = df[col].max()
    df[col] = (df[col] - col_min) / ((col_max - col_min) + 0.000000001)

    return df 

def create_csvdata(data_dict):
    raw_data = data_dict["hits"]["hits"]
    
    #benign_flows = [hit for hit in data
    #                if hit["_source"].get("suricata", {}).get("alert") in ({}, None)]
    
    #print(f"Found {len(benign_flows)} benign flows")
    #print(f"Found {len(data) - len(benign_flows)} alert flows")

    data = {
        "src_port": [],
        "dest_port": [],
        "bytes_toclient": [],
        "pkts_toclient": [],
        "bytes_toserver": [],
        "pkts_toserver": [],
        "direction": [],
        "duration": [],
        "protocol": [],
        "label": []
    }

    # figure out protocol col

    data = populate(raw_data, data) 

    # remap port numbers to categories for one hot encoding
    data = remap(data)

    # convert to pandas df
    df = pd.DataFrame(data)

    # one hot encode our dataframe
    df = one_hot(df)

    numeric_cols = ["bytes_toclient", "bytes_toserver", "pkts_toclient", "pkts_toserver", "duration"]
    # standardize bytes, pkts, duration
    for col in numeric_cols:
        df = min_max(df, col)

    # write csv
    df.to_csv("./data/data.csv", index=False)

def download_netflow_dataset():
    if len(sys.argv) != 3:
        print("ERROR: USERNAME and PASSWORD not specified.")
        sys.exit()
    
    username = sys.argv[1]
    password = sys.argv[2]
    
    # load the data   
    data_dict = load_data(username, password)
    
    # create csv of useful data
    create_csvdata(data_dict)

def stat_netflow_dataset():

    # Locals
    df = None
    labels = None
    size = 0

    aggregateInformation = [] # represents array of information related to columns
    # Format of entries in the format of:
    """
        {
            "column_name": str # represents the name of the column the entry represents
            "type": str # represents the type of data the column holds. "categorical" indicates values are 0 or 1 while "numerical" indicates values are any non-zero real number
            "information": ---- # holds information related to the column
        }

        If type is "categorical" information is a number, representing the total instances of the category occuring in the dataset

        If type is "numerical" information is a dictionary with aggregate information on the numerical data of the column in the form:
        {
            "avg":float,
            "min":float,
            "max":float
        }

    """

    nonCategoricalColumns = [   # Columns are assumed to be categorical unless specified to be numerical here
        'bytes_toclient',
        'pkts_toclient',
        'bytes_toserver',
        'pkts_toserver',
        'duration'
    ]


    df = pd.read_csv("./data/data.csv")
    labels = df.columns
    df = df.to_numpy()
    
    # For each label, initialize them in the aggregateInformation dict
    for i in range(0, len(labels)):
        if labels[i] in nonCategoricalColumns:
            aggregateInformation.append({
                "column_name":labels[i],
                "type":"numerical",
                "information":{
                    "avg":0,
                    "min":-1,
                    "max":-1
                }
            })
        else:
            aggregateInformation.append({
                "column_name":labels[i],
                "type":"categorical",
                "information":0
            })

    # Calculate aggregate information of each column
    #   Gather information on each column
    for row in df:
        size = size + 1
        for i in range(0,len(row)):

            if (aggregateInformation[i]["type"] == "categorical"):
                # Calculate sum
                aggregateInformation[i]["information"] = aggregateInformation[i]["information"] + row[i]

            if (aggregateInformation[i]["type"] == "numerical"):

                # Calculate sum (later to turn into avg)
                aggregateInformation[i]["information"]["avg"] = aggregateInformation[i]["information"]["avg"] + row[i]

                # Calculate min
                if (aggregateInformation[i]["information"]["min"] > row[i] or aggregateInformation[i]["information"]["min"] == -1):
                    aggregateInformation[i]["information"]["min"] = row[i]

                # Calculate max
                if (aggregateInformation[i]["information"]["max"] < row[i] or aggregateInformation[i]["information"]["max"] == -1):
                    aggregateInformation[i]["information"]["max"] = row[i]

    # Output results to output file in the data folder
    with open("./data/data_stat" + ".csv", "w+") as output_file:

        # output results in form "column_name, {information}" for each column (allowing a new line per column)
        for entry in aggregateInformation:
            
            output_file.write(entry["column_name"] + ",")

            if (entry["type"] == "numerical"):
                output_file.write("avg: " + str(entry["information"]["avg"]/size) + ",")
                output_file.write("min: " + str(entry["information"]["min"]) + ",")
                output_file.write("max: " + str(entry["information"]["max"]) + ",\n")

            if (entry["type"] == "categorical"):
                output_file.write("total instances: " + str(entry["information"]) + ",")
                output_file.write("appearance ratio: " + str(entry["information"]/size) + ",\n")




