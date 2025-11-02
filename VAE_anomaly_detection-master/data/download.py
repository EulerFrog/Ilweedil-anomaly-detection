import requests
from requests.auth import HTTPBasicAuth
import json
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import sys

if len(sys.argv) != 3:
    print("ERROR: USERNAME and PASSWORD not specified.")
    sys.exit()

username = sys.argv[1]
password = sys.argv[2]

# Suppress HTTPS warnings (because verify=False)
urllib3.disable_warnings(InsecureRequestWarning)

# OpenSearch URL
url = "https://140.160.1.66:9200/_search"

# Simple query: get first 10 documents from all indexes
query = {
    "query": { "match_all": {} },
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


breakpoint()
# Print the response
print(response.text)
