import requests
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')

# Etherscan API endpoint
BASE_URL = "https://api.etherscan.io/api"


def get_normal_transactions(address):
    url = f"{BASE_URL}?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if data["status"] != "1":
        return pd.DataFrame()
    return pd.DataFrame(data["result"])

def get_erc20_transactions(address):
    url = f"{BASE_URL}?module=account&action=tokentx&address={address}&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if data["status"] != "1":
        return pd.DataFrame()
    return pd.DataFrame(data["result"])

def main():
    print()
    return

if __name__ == "__main__":
    if not ETHERSCAN_API_KEY:
        print("Error: Etherscan API key not found in .env file")
    else:
        main()