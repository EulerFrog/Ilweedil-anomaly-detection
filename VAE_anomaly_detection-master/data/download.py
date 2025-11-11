import requests
from requests.auth import HTTPBasicAuth
import json
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import sys
import pandas as pd

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
        "size": 10
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
        raw_data = raw_data["_source"]

        breakpoint()
        for col in columns:
            if col == "direction":
                data[col].append(raw_data[col])
            else if col == "duration":
                duration = (int(raw_data["event"]["end"]) - int(raw_data["event"]["start"])) / 1000 # convert from ms to sec
                data[col].append(duration)
            else if col == "label":
                data[col].append(0 if not raw_data["suricata"][col]) else 1)
            else:
                data[col].append(raw_data["suricata"][col])
 
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

    for idx, src_port, dest_port in enumerate(zip(data["src_port"], data["dest_port"])):
        if str(src_port) in common_ports.keys():
            data["src_port"][idx] = common_ports[str(src_port)]
        else if src_port < 1024:
            data["src_port"][idx] = "public"
        else if src_port < 49152:
            data["src_port"][idx] = "private"
        else if src_port > 49151:
            data["src_port"][idx] = "dynamic"

        if str(dest_port) in common_ports.keys():
            data["dest_port"][idx] = common_ports[str(dest_port)]
        else if dest_port < 1024:
            data["dest_port"][idx] = "public"
        else if dest_port < 49152:
            data["dest_port"][idx] = "private"
        else if dest_port > 49151:
            data["dest_port"][idx] = "dynamic"

    return data

def create_csvdata(data):
    raw_data = data_dict["hits"]["hits"]
    
    #benign_flows = [hit for hit in data
    #                if hit["_source"].get("suricata", {}).get("alert") in ({}, None)]
    
    #print(f"Found {len(benign_flows)} benign flows")
    #print(f"Found {len(data) - len(benign_flows)} alert flows")

    breakpoint()

    data = {
        "src_port": [],
        "dest_port": [],
        "bytes_toclient": [],
        "pkts_toclient": [],
        "bytes_toserver": [],
        "pkts_toclient": [],
        "direction": [],
        "duration": [],
        "protocol": [],
        "label": []
    }

    # figure out protocol col

    data = populate(raw_data, data) 

    # one hot encode ports, direction, protocol
    # remap port numbers to categories for one hot encoding
    data = remap(data)

    df = pd.DataFrame(data)
    # standardize bytes, pkts, duration

    # write csv
    df = pd.DataFrame(data)
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
    create_csvdata(data_dict["hits"]["hits"])

 
if __name__ == "__main__":
    main()
