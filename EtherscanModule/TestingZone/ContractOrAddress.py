import os
import requests
import csv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('ETHERSCAN_API_KEY')  # Assuming the key in .env is ETHERSCAN_API_KEY

f = open("Address.csv","r")

lst = list(f)


def is_contract(address, api_key):
    url = f"https://api.etherscan.io/api?module=proxy&action=eth_getCode&address={address}&tag=latest&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        result = data.get('result', '')
        return 1 if result != '0x' else 0
    else:
        print(f"Error querying {address}: {response.status_code}")
        return 0  # Default to 0 on error, or handle differently as needed

def write():
    o=open("output.csv","w")
    o.write(lst[0])
    for i in range(1,len(lst)):
        l = lst[i].split(",")
        o.write(l[0]+","+l[1]+","+l[2]+ "," + str(is_contract(l[2],API_KEY)) + "\n")
        print(i-1)

write()