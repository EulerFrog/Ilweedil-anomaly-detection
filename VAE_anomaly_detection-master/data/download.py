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
    df.to_csv("data.csv", index=False)


def main():
    if len(sys.argv) != 3:
        print("ERROR: USERNAME and PASSWORD not specified.")
        sys.exit()
    
    username = sys.argv[1]
    password = sys.argv[2]
    
    # load the data   
    data_dict = load_data(username, password)
    
    # create csv of useful data
    create_csvdata(data_dict)

 
if __name__ == "__main__":
    main()
