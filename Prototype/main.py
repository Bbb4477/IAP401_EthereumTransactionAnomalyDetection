import argparse
import sys
import os
import pandas as pd
import numpy as np
import pickle
import requests
import time
from collections import defaultdict
import json
from dotenv import load_dotenv
import re
import shap
import concurrent.futures
import threading
# import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')

# Etherscan API endpoint (corrected to standard URL)
BASE_URL = "https://api.etherscan.io/v2/api"

def is_contract(addr, api_key, timeout=None):
    """
    Check if an address is a contract using Etherscan API.
    """
    if not addr or addr == '0x':
        return False
    params = {
        'chainid': '1',
        'module': 'proxy',
        'action': 'eth_getCode',
        'address': addr,
        'tag': 'latest',
        'apikey': api_key
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=timeout)
        if resp.status_code != 200:
            return False
        data = resp.json()
        code = data.get('result')
        return len(code) > 2
    except requests.Timeout:
        return None
    except Exception:
        return False

def get_txlist(address, api_key, action='txlist'):
    """
    Fetch transaction list from Etherscan API.
    Assumes less than 10000 tx; for more, pagination needed.
    """
    params = {
        'chainid': '1',
        'module': 'account',
        'action': action,
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'page': 1,
        'offset': 10000,
        'sort': 'asc',
        'apikey': api_key
    }
    try:
        resp = requests.get(BASE_URL, params=params)
        time.sleep(2)  # Rate limit
        if resp.status_code != 200:
            return []
        data = resp.json()
        if data.get('status') == '0':
            print(f"API Error for {action}: {data.get('message', 'Unknown')}")
            return []
        t=""
        if action == 'tokentx':
            t = "ERC20 "
        print(f"{t}Transaction crawl Completed")
        return data['result']
    except Exception as e:
        print(f"Error fetching {action}: {e}")
        return []

def avg_min_erc(txs_list):
    """
    Calculate average minutes between ERC20 transactions.
    """
    if len(txs_list) < 2:
        return 0.0
    times = sorted(int(tx['timeStamp']) for tx in txs_list)
    diffs = [(times[i+1] - times[i]) / 60.0 for i in range(len(times)-1)]
    return sum(diffs) / len(diffs) if diffs else 0.0

# def save_to_csv(df, output_path='out.csv', mode='w'):
#     """
#     Save a DataFrame to a CSV file, appending or overwriting based on mode.
#
#     Parameters:
#     df (pandas.DataFrame): The DataFrame to save.
#     output_path (str): Path to the output CSV file. Defaults to 'out.csv'.
#     mode (str): Write mode ('w' for overwrite, 'a' for append). Defaults to 'w'.
#     """
#     try:
#         if mode == 'a' and os.path.exists(output_path):
#             # Load existing CSV and append
#             existing_df = pd.read_csv(output_path)
#             # Remove duplicate address if exists
#             existing_df = existing_df[existing_df['Address'] != df['Address'].iloc[0]]
#             df = pd.concat([existing_df, df], ignore_index=True)
#         df.to_csv(output_path, index=False)
#         print(f"Data successfully saved to '{output_path}'")
#     except Exception as e:
#         print(f"Error saving to CSV: {e}")

# def save_to_csv(df, output_path='out.csv', mode='w'):
#     """
#     Save a DataFrame to a CSV file, appending or overwriting based on mode.
#     Ensures no extra 'Unnamed' columns are added.
#     """
#     try:
#         # Drop any 'Unnamed' columns from input DataFrame
#         df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
#
#         if mode == 'a' and os.path.exists(output_path):
#             # Read existing CSV, drop 'Unnamed' columns, and filter out duplicate address
#             existing_df = pd.read_csv(output_path)
#             existing_df = existing_df.loc[:, ~existing_df.columns.str.contains('^Unnamed', na=False)]
#             existing_df = existing_df[existing_df['Address'] != df['Address'].iloc[0]]
#             df = pd.concat([existing_df, df], ignore_index=True)
#
#         # Save DataFrame without index to prevent 'Unnamed' columns
#         df.to_csv(output_path, index=False)
#         print(f"Data successfully saved to '{output_path}'")
#     except Exception as e:
#         print(f"Error saving to CSV: {e}")

def save_to_csv(df, output_path='out.csv', mode='w'):
    """
    Save DataFrame to CSV:
    - Only ASCII characters allowed in string columns
    - Any non-ASCII character is replaced with '' (removed)
    - Always saves in clean UTF-8
    - Prevents encoding errors forever
    """
    import pandas as pd
    import os

    def to_ascii_only(text):
        if isinstance(text, str):
            return ''.join(c if ord(c) < 128 else '' for c in text)  # Keep only ASCII < 128
        return text

    try:
        # Apply ASCII-only cleaning to all string/object columns
        df = df.copy()  # Avoid modifying original
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(to_ascii_only)

        # Remove any Unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]

        if mode == 'a' and os.path.exists(output_path):
            # Load existing file safely (with ASCII cleanup too)
            existing_df = pd.read_csv(output_path, encoding='utf-8', dtype=str)  # dtype=str to preserve strings
            existing_df = existing_df.loc[:, ~existing_df.columns.str.contains('^Unnamed', na=False)]

            # Clean existing data too
            for col in existing_df.select_dtypes(include=['object']).columns:
                existing_df[col] = existing_df[col].apply(to_ascii_only)

            # Remove duplicate Address
            existing_df = existing_df[existing_df['Address'] != df['Address'].iloc[0]]
            df = pd.concat([existing_df, df], ignore_index=True)

        # Save with pure UTF-8 and no index
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Data saved to '{output_path}' (non-ASCII characters removed)")

    except Exception as e:
        print(f"Error saving CSV: {e}")

def wei_to_eth(v_str):
    return int(v_str) / 1e18 if v_str else 0.0

def adjusted_value(tx):
    if not tx.get('value'):
        return 0.0
    try:
        val = int(tx['value'])
    except ValueError:
        print(f"Warning: Invalid 'value' in tx: {tx.get('hash')}, skipping.")
        return 0.0

    decimal_str = tx.get('tokenDecimal')
    try:
        decimals = int(decimal_str) if decimal_str else 18
    except ValueError:
        print(
            f"Warning: Invalid 'tokenDecimal' '{decimal_str}' for token {tx.get('tokenName')} in tx: {tx.get('hash')}, defaulting to 18.")
        decimals = 18

    try:
        return val / (10 ** decimals)
    except OverflowError:
        # Rare case for extremely large exponents, but unlikely with decimals ~0-18
        return float(val) / (10.0 ** decimals)

def calc_time_diff(txs):
    if not txs:
        return 0.0
    timestamps = [int(tx['timeStamp']) for tx in txs]
    return (max(timestamps) - min(timestamps)) / 60.0

def calc_avg_min_between(txs):
    if len(txs) < 2:
        return 0.0
    times = sorted(int(tx['timeStamp']) for tx in txs)
    diffs = [(times[i+1] - times[i]) / 60.0 for i in range(len(times)-1)]
    return sum(diffs) / len(diffs) if diffs else 0.0

def calc_sent_received_counts(txs, address):
    sent_txs = [tx for tx in txs if tx['from'].lower() == address.lower()]
    rec_txs = [tx for tx in txs if tx['to'] and tx['to'].lower() == address.lower()]
    return len(sent_txs), len(rec_txs)

def calc_created_contracts(txs, address):
    sent_txs = [tx for tx in txs if tx['from'].lower() == address.lower()]
    return sum(1 for tx in sent_txs if tx['contractAddress'] != '')

def calc_unique_addresses(txs, address):
    sent_txs = [tx for tx in txs if tx['from'].lower() == address.lower()]
    rec_txs = [tx for tx in txs if tx['to'] and tx['to'].lower() == address.lower()]
    return len(set(tx['from'].lower() for tx in rec_txs)), len(set(tx['to'].lower() for tx in sent_txs if tx['to']))

def calc_eth_values(txs, address):
    rec_txs = [tx for tx in txs if tx['to'] and tx['to'].lower() == address.lower()]
    sent_txs = [tx for tx in txs if tx['from'].lower() == address.lower()]
    rec_vals = [wei_to_eth(tx['value']) for tx in rec_txs]
    sent_vals = [wei_to_eth(tx['value']) for tx in sent_txs]
    min_rec = min(rec_vals) if rec_vals else 0.0
    max_rec = max(rec_vals) if rec_vals else 0.0
    avg_rec = sum(rec_vals) / len(rec_vals) if rec_vals else 0.0
    min_sent = min(sent_vals) if sent_vals else 0.0
    max_sent = max(sent_vals) if sent_vals else 0.0
    avg_sent = sum(sent_vals) / len(sent_vals) if sent_vals else 0.0
    total_sent = sum(sent_vals)
    total_rec = sum(rec_vals)
    return min_rec, max_rec, avg_rec, min_sent, max_sent, avg_sent, total_sent, total_rec

def calc_eth_contract_values(txs, address, contract_cache):
    sent_txs = [tx for tx in txs if tx['from'].lower() == address.lower()]
    contract_sent_vals = []
    for tx in sent_txs:
        if tx['contractAddress'] != '':
            contract_sent_vals.append(wei_to_eth(tx['value']))
            continue
        to = tx['to'].lower() if tx['to'] else ''
        if not to:
            continue
        if tx['input'] != '0x' or contract_cache.get(to, False):
            contract_sent_vals.append(wei_to_eth(tx['value']))
    min_sent_contract = min(contract_sent_vals) if contract_sent_vals else 0.0
    max_sent_contract = max(contract_sent_vals) if contract_sent_vals else 0.0
    avg_sent_contract = sum(contract_sent_vals) / len(contract_sent_vals) if contract_sent_vals else 0.0
    total_sent_contracts = sum(contract_sent_vals)
    return min_sent_contract, max_sent_contract, avg_sent_contract, total_sent_contracts

def calc_total_transactions(sent_count, rec_count):
    return sent_count + rec_count

def calc_total_balance(total_sent, total_rec):
    return total_rec - total_sent

def calc_erc20_counts(erc_txs, address):
    sent_erc_txs = [tx for tx in erc_txs if tx['from'].lower() == address.lower()]
    rec_erc_txs = [tx for tx in erc_txs if tx['to'].lower() == address.lower()]
    return len(sent_erc_txs) + len(rec_erc_txs)

def calc_erc20_values(erc_txs, address):
    sent_erc_txs = [tx for tx in erc_txs if tx['from'].lower() == address.lower()]
    rec_erc_txs = [tx for tx in erc_txs if tx['to'].lower() == address.lower()]
    rec_vals = [adjusted_value(tx) for tx in rec_erc_txs]
    sent_vals = [adjusted_value(tx) for tx in sent_erc_txs]
    total_rec = sum(rec_vals) if rec_vals else 0.0
    total_sent = sum(sent_vals) if sent_vals else 0.0
    min_rec = min(rec_vals) if rec_vals else 0.0
    max_rec = max(rec_vals) if rec_vals else 0.0
    avg_rec = total_rec / len(rec_vals) if rec_vals else 0.0
    min_sent = min(sent_vals) if sent_vals else 0.0
    max_sent = max(sent_vals) if sent_vals else 0.0
    avg_sent = total_sent / len(sent_vals) if sent_vals else 0.0
    return total_rec, total_sent, min_rec, max_rec, avg_rec, min_sent, max_sent, avg_sent

def calc_erc20_contract_values(erc_txs, address, contract_cache):
    sent_erc_txs = [tx for tx in erc_txs if tx['from'].lower() == address.lower()]
    contract_vals = [adjusted_value(tx) for tx in sent_erc_txs if tx['to'] and contract_cache.get(tx['to'].lower(), False)]
    total_contract = sum(contract_vals) if contract_vals else 0.0
    min_contract = min(contract_vals) if contract_vals else 0.0
    max_contract = max(contract_vals) if contract_vals else 0.0
    avg_contract = total_contract / len(contract_vals) if contract_vals else 0.0
    return total_contract, min_contract, max_contract, avg_contract

def calc_erc20_unique_addresses(erc_txs, address):
    sent_erc_txs = [tx for tx in erc_txs if tx['from'].lower() == address.lower()]
    rec_erc_txs = [tx for tx in erc_txs if tx['to'].lower() == address.lower()]
    uniq_sent_addr = len(set(tx['to'].lower() for tx in sent_erc_txs if tx['to']))
    uniq_rec_addr = len(set(tx['from'].lower() for tx in rec_erc_txs))
    uniq_sent_contract_addr = len(set(tx['contractAddress'].lower() for tx in sent_erc_txs if tx['contractAddress']))
    uniq_rec_contract_addr = len(set(tx['contractAddress'].lower() for tx in rec_erc_txs if tx['contractAddress']))
    return uniq_sent_addr, uniq_rec_addr, uniq_sent_contract_addr, uniq_rec_contract_addr

def calc_erc20_time_metrics(erc_txs, address, contract_cache):
    sent_erc_txs = [tx for tx in erc_txs if tx['from'].lower() == address.lower()]
    rec_erc_txs = [tx for tx in erc_txs if tx['to'].lower() == address.lower()]
    avg_sent_time = calc_avg_min_between(sent_erc_txs)
    avg_rec_time = calc_avg_min_between(rec_erc_txs)
    avg_rec2_time = 0.0
    if len(rec_erc_txs) >= 3:
        times = sorted(int(tx['timeStamp']) for tx in rec_erc_txs)
        diffs = [(times[i+1] - times[i]) / 60.0 for i in range(1, len(times)-1)]
        avg_rec2_time = sum(diffs) / len(diffs) if diffs else 0.0
    unique_parties = set(tx['from'].lower() for tx in erc_txs) | set(tx['to'].lower() for tx in erc_txs if tx['to'])
    unique_parties.discard(address.lower())
    contract_txs = [tx for tx in erc_txs if contract_cache.get(tx['from'].lower(), False) or (tx['to'] and contract_cache.get(tx['to'].lower(), False))]
    avg_contract_time = calc_avg_min_between(contract_txs)
    return avg_sent_time, avg_rec_time, avg_rec2_time, avg_contract_time

def calc_erc20_token_names(erc_txs, address):
    sent_erc_txs = [tx for tx in erc_txs if tx['from'].lower() == address.lower()]
    rec_erc_txs = [tx for tx in erc_txs if tx['to'].lower() == address.lower()]
    uniq_sent_tokens = len(set(tx['tokenName'] for tx in sent_erc_txs)) if sent_erc_txs else 0.0
    uniq_rec_tokens = len(set(tx['tokenName'] for tx in rec_erc_txs)) if rec_erc_txs else 0.0
    return uniq_sent_tokens, uniq_rec_tokens

def calc_erc20_most_token_types(erc_txs, address):
    sent_erc_txs = [tx for tx in erc_txs if tx['from'].lower() == address.lower()]
    rec_erc_txs = [tx for tx in erc_txs if tx['to'].lower() == address.lower()]
    sent_token_sum = defaultdict(float)
    for tx in sent_erc_txs:
        name = tx.get('tokenName', '')
        sent_token_sum[name] += adjusted_value(tx)
    most_sent_type = max(sent_token_sum, key=sent_token_sum.get) if sent_token_sum else ''
    rec_token_sum = defaultdict(float)
    for tx in rec_erc_txs:
        name = tx.get('tokenName', '')
        rec_token_sum[name] += adjusted_value(tx)
    most_rec_type = max(rec_token_sum, key=rec_token_sum.get) if rec_token_sum else ''
    return most_sent_type, most_rec_type

# def save_raw_data(txs, erc_txs, contract_cache, address):
#     os.makedirs('raw', exist_ok=True)
#     pd.DataFrame(txs).to_csv(f'raw/transactions-{address.lower()}.csv', index=False)
#     pd.DataFrame(erc_txs).to_csv(f'raw/erc20_transactions-{address.lower()}.csv', index=False)
#     with open(f'raw/contracts-{address.lower()}.json', 'w') as f:
#         json.dump(contract_cache, f)
#     print(f"Raw data saved to raw/ for address {address}: transactions-{address.lower()}.csv, erc20_transactions-{address.lower()}.csv, contracts-{address.lower()}.json")


# def build_contract_cache(unique_addrs, api_key):
#     """
#     Build contract cache by checking local JSON file and querying Etherscan only for missing addresses.
#     Updates the JSON file with new contract statuses as True/False.
#     """
#     os.makedirs('raw', exist_ok=True)
#     contract_cache_file = 'raw/contracts-total.json'
#     contract_cache = {}
#
#     # Load existing contract cache from JSON if it exists
#     if os.path.exists(contract_cache_file):
#         try:
#             with open(contract_cache_file, 'r') as f:
#                 contract_cache = json.load(f)
#             # Ensure all keys are lowercase
#             contract_cache = {k.lower(): v for k, v in contract_cache.items()}
#             print(f"Loaded {len(contract_cache)} contract statuses from {contract_cache_file}")
#         except Exception as e:
#             print(f"Error loading {contract_cache_file}: {e}, starting with empty cache")
#
#     # Identify addresses not in cache
#     new_addrs = [addr for addr in unique_addrs if addr.lower() not in contract_cache]
#     print(f"Checking {len(new_addrs)} new addresses via Etherscan API")
#
#     # Query Etherscan for new addresses
#     for addr in new_addrs:
#         contract_cache[addr.lower()] = is_contract(addr, api_key)
#
#     # Save updated contract cache to JSON
#     try:
#         with open(contract_cache_file, 'w') as f:
#             json.dump(contract_cache, f)
#         print(f"Updated contract cache saved to {contract_cache_file}")
#     except Exception as e:
#         print(f"Error saving {contract_cache_file}: {e}")
#
#     return contract_cache

def build_contract_cache(unique_addrs, api_key_input):
    """
    Build contract cache by checking local JSON file and querying Etherscan only for missing addresses.
    Uses multiple API keys in parallel, updates JSON file every 1000 new calls with True/False statuses.
    Implements Fast First Wave + Immediate Serial Fallback for reliability and speed.
    """
    os.makedirs('raw', exist_ok=True)
    contract_cache_file = 'raw/contracts-total.json'
    contract_cache = {}

    # Load existing contract cache from JSON if it exists
    if os.path.exists(contract_cache_file):
        try:
            with open(contract_cache_file, 'r') as f:
                contract_cache = json.load(f)
            contract_cache = {k.lower(): v for k, v in contract_cache.items()}
            print(f"Loaded {len(contract_cache)} contract statuses from {contract_cache_file}")
        except Exception as e:
            print(f"Error loading {contract_cache_file}: {e}, starting with empty cache")

    # Identify addresses not in cache
    new_addrs = [addr for addr in unique_addrs if addr.lower() not in contract_cache]
    api_keys = load_api_keys(api_key_input)
    print(
        f"Checking {len(new_addrs)} ({len(unique_addrs)}) new addresses via Etherscan API ({len(api_keys)} keys in use)")

    if not new_addrs:
        return contract_cache

    # WAVE 1: Aggressive parallel with semaphore and timeouts
    max_concurrent = len(api_keys) * 4  # Safe: 4 per key
    sem = threading.Semaphore(max_concurrent)

    def check_contract(addr, key):
        with sem:
            time.sleep(0.22)  # Respect ~4.5 calls/sec per key
            result = is_contract(addr, key, timeout=60)
            return addr.lower(), result

    failed = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        for i, addr in enumerate(new_addrs):
            key = api_keys[i % len(api_keys)]
            future = executor.submit(check_contract, addr, key)
            futures[future] = addr

        for future in concurrent.futures.as_completed(futures):
            addr = futures[future]
            try:
                addr_lower, result = future.result()
                if result is not None:
                    contract_cache[addr_lower] = result
                    completed += 1
                    if completed % 1000 == 0 or completed == len(new_addrs):
                        try:
                            with open(contract_cache_file, 'w') as f:
                                json.dump(contract_cache, f)
                            print(f"{completed}/{len(new_addrs)} calls complete, updated {contract_cache_file}")
                        except Exception as e:
                            print(f"Error saving {contract_cache_file} at {completed} calls: {e}")
                else:
                    failed.append(addr)
            except Exception as e:
                print(f"Error for {addr}: {e}")
                failed.append(addr)

    print(f"Wave 1 complete. {len(failed)} failed/timeout → starting serial fallback")

    # SERIAL FALLBACK: Round-robin keys, no sleep, longer timeout
    for i, addr in enumerate(failed):
        key = api_keys[i % len(api_keys)]
        result = is_contract(addr, key, timeout=90)
        addr_lower = addr.lower()
        if result is not None:
            contract_cache[addr_lower] = result
        else:
            contract_cache[addr_lower] = False
            print(f"Even serial timeout for {addr}, marking as False")
        print(f"Serial fallback {i+1}/{len(failed)}: {addr[:10]}... OK")

    # Final save after serial
    try:
        with open(contract_cache_file, 'w') as f:
            json.dump(contract_cache, f)
        print(f"Final update: {contract_cache_file}")
    except Exception as e:
        print(f"Error saving {contract_cache_file} at end: {e}")

    print("Completed Crawling")

    return contract_cache

def load_api_keys(api_key_input=None):
    """
    Load Etherscan API keys from .env or input string.
    Returns a list of keys, falling back to single key if none provided.
    """
    api_keys = []
    if api_key_input:
        # Handle comma-separated input
        api_keys = [key.strip() for key in api_key_input.split(',') if key.strip()]
    if not api_keys:
        # Fallback to .env
        load_dotenv()
        env_key = os.getenv('ETHERSCAN_API_KEY')
        if env_key:
            api_keys = [key.strip() for key in env_key.split(',') if key.strip()]
    if not api_keys:
        print("Error: No valid API keys provided in input or .env")
        sys.exit(1)
    return api_keys

def get_transaction_and_metadata(address, api_key):
    # Fetch normal transactions and save immediately
    key_list = load_api_keys(api_key)
    first_key = key_list[-1] if key_list else None
    if not first_key:
        print("Error: No valid API key available")
        sys.exit(1)
    txs = get_txlist(address, first_key, 'txlist')
    os.makedirs('raw', exist_ok=True)
    pd.DataFrame(txs).to_csv(f'raw/transactions-{address.lower()}.csv', index=False)
    print(f"Raw transactions saved to raw/transactions-{address.lower()}.csv")

    # Fetch ERC20 transactions and save immediately
    erc_txs = get_txlist(address, first_key, 'tokentx')
    pd.DataFrame(erc_txs).to_csv(f'raw/erc20_transactions-{address.lower()}.csv', index=False)
    print(f"Raw ERC20 transactions saved to raw/erc20_transactions-{address.lower()}.csv")

    tempContract = 1 if is_contract(address, first_key) else 0

    # Build contract cache
    unique_addrs = set()
    for tx in txs:
        unique_addrs.add(tx.get('from', '').lower())
        unique_addrs.add(tx.get('to', '').lower())
        unique_addrs.add(tx.get('contractAddress', '').lower())
    for tx in erc_txs:
        unique_addrs.add(tx.get('from', '').lower())
        unique_addrs.add(tx.get('to', '').lower())
        unique_addrs.add(tx.get('contractAddress', '').lower())
    unique_addrs.discard('')

    # contract_cache = {addr: is_contract(addr, api_key) for addr in unique_addrs}
    contract_cache = build_contract_cache(unique_addrs, api_key)

    # Save raw data
    # save_raw_data(txs, erc_txs, contract_cache, address)

    # Save contract cache for this address's unique addresses only
    address_specific_cache = {addr: contract_cache[addr] for addr in unique_addrs if addr in contract_cache}
    with open(f'raw/contracts-{address.lower()}.json', 'w') as f:
        json.dump(address_specific_cache, f)
    print(f"Contract cache saved to raw/contracts-{address.lower()}.json")

    if not txs:
        return pd.DataFrame([{
            'Unnamed: 0': 0,
            'Index': 1,
            'Address': address,
            'FLAG': 2,
            'Contract': tempContract,
            'Avg_min_between_sent_tnx': 0.0,
            'Avg_min_between_received_tnx': 0.0,
            'Time_Diff_between_first_and_last_(Mins)': 0.0,
            'Sent_tnx': 0,
            'Received_Tnx': 0,
            'Number_of_Created_Contracts': 0,
            'Unique_Received_From_Addresses': 0,
            'Unique_Sent_To_Addresses': 0,
            'min_value_received': 0.0,
            'max_value_received': 0.0,
            'avg_val_received': 0.0,
            'min_val_sent': 0.0,
            'max_val_sent': 0.0,
            'avg_val_sent': 0.0,
            'min_value_sent_to_contract': 0.0,
            'max_val_sent_to_contract': 0.0,
            'avg_value_sent_to_contract': 0.0,
            'total_transactions_(including_tnx_to_create_contract': 0,
            'total_Ether_sent': 0.0,
            'total_ether_received': 0.0,
            'total_ether_sent_contracts': 0.0,
            'total_ether_balance': 0.0,
            'Total_ERC20_tnxs': 0,
            'ERC20_total_Ether_received': 0.0,
            'ERC20_total_ether_sent': 0.0,
            'ERC20_total_Ether_sent_contract': 0.0,
            'ERC20_uniq_sent_addr': 0.0,
            'ERC20_uniq_rec_addr': 0.0,
            'ERC20_uniq_sent_addr.1': 0.0,
            'ERC20_uniq_rec_contract_addr': 0.0,
            'ERC20_avg_time_between_sent_tnx': 0.0,
            'ERC20_avg_time_between_rec_tnx': 0.0,
            'ERC20_avg_time_between_rec_2_tnx': 0.0,
            'ERC20_avg_time_between_contract_tnx': 0.0,
            'ERC20_min_val_rec': 0.0,
            'ERC20_max_val_rec': 0.0,
            'ERC20_avg_val_rec': 0.0,
            'ERC20_min_val_sent': 0.0,
            'ERC20_max_val_sent': 0.0,
            'ERC20_avg_val_sent': 0.0,
            'ERC20_min_val_sent_contract': 0.0,
            'ERC20_max_val_sent_contract': 0.0,
            'ERC20_avg_val_sent_contract': 0.0,
            'ERC20_uniq_sent_token_name': 0.0,
            'ERC20_uniq_rec_token_name': 0.0,
            'ERC20_most_sent_token_type': '',
            'ERC20_most_rec_token_type': ''
        }])

    time_diff = calc_time_diff(txs)
    print("time_diff: ", time_diff)

    sent_count, rec_count = calc_sent_received_counts(txs, address)
    print("sent_count: ", sent_count, " rec_count: ", rec_count)

    avg_sent_between = calc_avg_min_between([tx for tx in txs if tx['from'].lower() == address.lower()])
    print("avg_sent_between: ", avg_sent_between)

    avg_rec_between = calc_avg_min_between([tx for tx in txs if tx['to'] and tx['to'].lower() == address.lower()])
    print("avg_rec_between: ", avg_rec_between)

    created_count = calc_created_contracts(txs, address)
    print("created_count: ", created_count)

    unique_rec_from, unique_sent_to = calc_unique_addresses(txs, address)
    print("unique_rec_from: ", unique_rec_from, " unique_sent_to: ", unique_sent_to)

    min_rec, max_rec, avg_rec, min_sent, max_sent, avg_sent, total_sent, total_rec = calc_eth_values(txs, address)
    print("min_rec: ", min_rec, " max_rec: ", max_rec, " avg_rec: ", avg_rec,
          " min_sent: ", min_sent, " max_sent: ", max_sent, " avg_sent: ", avg_sent,
          " total_sent: ", total_sent, " total_rec: ", total_rec)

    min_sent_contract, max_sent_contract, avg_sent_contract, total_sent_contracts = calc_eth_contract_values(txs, address, contract_cache)
    print("min_sent_contract: ", min_sent_contract, " max_sent_contract: ", max_sent_contract,
          " avg_sent_contract: ", avg_sent_contract, " total_sent_contracts: ", total_sent_contracts)

    total_tx = calc_total_transactions(sent_count, rec_count)
    print("total_tx: ", total_tx)

    total_balance = calc_total_balance(total_sent, total_rec)
    print("total_balance: ", total_balance)

    total_erc_tnxs = calc_erc20_counts(erc_txs, address)
    print("total_erc_tnxs: ", total_erc_tnxs)

    erc_total_rec, erc_total_sent, erc_min_rec, erc_max_rec, erc_avg_rec, erc_min_sent, erc_max_sent, erc_avg_sent = calc_erc20_values(erc_txs, address)
    print("erc_total_rec: ", erc_total_rec, " erc_total_sent: ", erc_total_sent,
          " erc_min_rec: ", erc_min_rec, " erc_max_rec: ", erc_max_rec, " erc_avg_rec: ", erc_avg_rec,
          " erc_min_sent: ", erc_min_sent, " erc_max_sent: ", erc_max_sent, " erc_avg_sent: ", erc_avg_sent)

    erc_total_sent_contract, erc_min_sent_contract, erc_max_sent_contract, erc_avg_sent_contract = calc_erc20_contract_values(erc_txs, address, contract_cache)
    print("erc_total_sent_contract: ", erc_total_sent_contract,
          " erc_min_sent_contract: ", erc_min_sent_contract,
          " erc_max_sent_contract: ", erc_max_sent_contract,
          " erc_avg_sent_contract: ", erc_avg_sent_contract)

    erc_uniq_sent_addr, erc_uniq_rec_addr, erc_uniq_sent_addr1, erc_uniq_rec_contract = calc_erc20_unique_addresses(erc_txs, address)
    print("erc_uniq_sent_addr: ", erc_uniq_sent_addr, " erc_uniq_rec_addr: ", erc_uniq_rec_addr,
          " erc_uniq_sent_addr1: ", erc_uniq_sent_addr1, " erc_uniq_rec_contract: ", erc_uniq_rec_contract)

    erc_avg_sent_time, erc_avg_rec_time, erc_avg_rec2_time, erc_avg_contract_time = calc_erc20_time_metrics(erc_txs, address, contract_cache)
    print("erc_avg_sent_time: ", erc_avg_sent_time, " erc_avg_rec_time: ", erc_avg_rec_time,
          " erc_avg_rec2_time: ", erc_avg_rec2_time, " erc_avg_contract_time: ", erc_avg_contract_time)

    erc_uniq_sent_token, erc_uniq_rec_token = calc_erc20_token_names(erc_txs, address)
    print("erc_uniq_sent_token: ", erc_uniq_sent_token, " erc_uniq_rec_token: ", erc_uniq_rec_token)

    most_sent_type, most_rec_type = calc_erc20_most_token_types(erc_txs, address)
    print("most_sent_type: ", most_sent_type, " most_rec_type: ", most_rec_type)

    # FirstContract = 0
    # if is_contract(address, api_key):
    #     FirstContract = 1

    df=pd.DataFrame([{
        'Unnamed: 0': 0,
        'Index': 1,
        'Address': address,
        'FLAG': 2,
        'Contract': tempContract,
        'Avg_min_between_sent_tnx': avg_sent_between,
        'Avg_min_between_received_tnx': avg_rec_between,
        'Time_Diff_between_first_and_last_(Mins)': time_diff,
        'Sent_tnx': sent_count,
        'Received_Tnx': rec_count,
        'Number_of_Created_Contracts': created_count + tempContract,
        'Unique_Received_From_Addresses': unique_rec_from,
        'Unique_Sent_To_Addresses': unique_sent_to,
        'min_value_received': min_rec,
        'max_value_received': max_rec,
        'avg_val_received': avg_rec,
        'min_val_sent': min_sent,
        'max_val_sent': max_sent,
        'avg_val_sent': avg_sent,
        'min_value_sent_to_contract': min_sent_contract,
        'max_val_sent_to_contract': max_sent_contract,
        'avg_value_sent_to_contract': avg_sent_contract,
        'total_transactions_(including_tnx_to_create_contract': total_tx + tempContract,
        'total_Ether_sent': total_sent,
        'total_ether_received': total_rec,
        'total_ether_sent_contracts': total_sent_contracts,
        'total_ether_balance': total_balance,
        'Total_ERC20_tnxs': total_erc_tnxs,
        'ERC20_total_Ether_received': erc_total_rec,
        'ERC20_total_ether_sent': erc_total_sent,
        'ERC20_total_Ether_sent_contract': erc_total_sent_contract,
        'ERC20_uniq_sent_addr': float(erc_uniq_sent_addr),
        'ERC20_uniq_rec_addr': float(erc_uniq_rec_addr),
        'ERC20_uniq_sent_addr.1': float(erc_uniq_sent_addr1),
        'ERC20_uniq_rec_contract_addr': float(erc_uniq_rec_contract),
        'ERC20_avg_time_between_sent_tnx': erc_avg_sent_time,
        'ERC20_avg_time_between_rec_tnx': erc_avg_rec_time,
        'ERC20_avg_time_between_rec_2_tnx': erc_avg_rec2_time,
        'ERC20_avg_time_between_contract_tnx': erc_avg_contract_time,
        'ERC20_min_val_rec': erc_min_rec,
        'ERC20_max_val_rec': erc_max_rec,
        'ERC20_avg_val_rec': erc_avg_rec,
        'ERC20_min_val_sent': erc_min_sent,
        'ERC20_max_val_sent': erc_max_sent,
        'ERC20_avg_val_sent': erc_avg_sent,
        'ERC20_min_val_sent_contract': erc_min_sent_contract,
        'ERC20_max_val_sent_contract': erc_max_sent_contract,
        'ERC20_avg_val_sent_contract': erc_avg_sent_contract,
        'ERC20_uniq_sent_token_name': erc_uniq_sent_token,
        'ERC20_uniq_rec_token_name': erc_uniq_rec_token,
        'ERC20_most_sent_token_type': most_sent_type,
        'ERC20_most_rec_token_type': most_rec_type
    }])
    # Handle infinite values with 0
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # Clip extremely large values to prevent overflow
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=-1e18, upper=1e18)

    # Verify no infinite or excessively large values
    if np.isinf(df[numeric_cols].values).any():
        print("WARNING: Infinite values still present!")
    elif (df[numeric_cols].abs() > 1e18).any().any():
        print("WARNING: Excessively large values still present!")
    else:
        print("✓ No infinite or excessively large values in data.")

    return df
# Prediction functions from the predict module

def process_offline_address(address):
    """
    Process already-downloaded raw data from raw/ folder.
    Reconstructs the same DataFrame as get_transaction_and_metadata() but offline.
    """
    addr_lower = address.lower()
    tx_path = f'raw/transactions-{addr_lower}.csv'
    erc_path = f'raw/erc20_transactions-{addr_lower}.csv'
    contract_cache_path = f'raw/contracts-{addr_lower}.json'

    print(f"Loading offline data for {address}...")

    if not os.path.exists(tx_path):
        print(f"Missing: {tx_path}")
        return None
    if not os.path.exists(erc_path):
        print(f"Missing: {erc_path}")
        return None

    txs = pd.read_csv(tx_path).to_dict('records')
    erc_txs = pd.read_csv(erc_path).to_dict('records')

    # Load contract cache
    contract_cache = {}
    if os.path.exists(contract_cache_path):
        with open(contract_cache_path, 'r') as f:
            contract_cache = json.load(f)
        contract_cache = {k.lower(): v for k, v in contract_cache.items()}
    else:
        print(f"Warning: No contract cache found: {contract_cache_path}")
        # Fallback: try global cache
        global_cache = 'raw/contracts-total.json'
        if os.path.exists(global_cache):
            with open(global_cache, 'r') as f:
                global_data = json.load(f)
                contract_cache = {k.lower(): v for k, v in global_data.items()}

    tempContract = 1 if contract_cache.get(addr_lower, False) else 0

    # Reuse all your existing calc_* functions
    # (They are already pure and work with list of dicts)

    if not txs:
        return pd.DataFrame([{
            'Unnamed: 0': 0,
            'Index': 1,
            'Address': address,
            'FLAG': 2,
            'Contract': tempContract,
            'Avg_min_between_sent_tnx': 0.0,
            'Avg_min_between_received_tnx': 0.0,
            'Time_Diff_between_first_and_last_(Mins)': 0.0,
            'Sent_tnx': 0,
            'Received_Tnx': 0,
            'Number_of_Created_Contracts': 0,
            'Unique_Received_From_Addresses': 0,
            'Unique_Sent_To_Addresses': 0,
            'min_value_received': 0.0,
            'max_value_received': 0.0,
            'avg_val_received': 0.0,
            'min_val_sent': 0.0,
            'max_val_sent': 0.0,
            'avg_val_sent': 0.0,
            'min_value_sent_to_contract': 0.0,
            'max_val_sent_to_contract': 0.0,
            'avg_value_sent_to_contract': 0.0,
            'total_transactions_(including_tnx_to_create_contract': 0,
            'total_Ether_sent': 0.0,
            'total_ether_received': 0.0,
            'total_ether_sent_contracts': 0.0,
            'total_ether_balance': 0.0,
            'Total_ERC20_tnxs': 0,
            'ERC20_total_Ether_received': 0.0,
            'ERC20_total_ether_sent': 0.0,
            'ERC20_total_Ether_sent_contract': 0.0,
            'ERC20_uniq_sent_addr': 0.0,
            'ERC20_uniq_rec_addr': 0.0,
            'ERC20_uniq_sent_addr.1': 0.0,
            'ERC20_uniq_rec_contract_addr': 0.0,
            'ERC20_avg_time_between_sent_tnx': 0.0,
            'ERC20_avg_time_between_rec_tnx': 0.0,
            'ERC20_avg_time_between_rec_2_tnx': 0.0,
            'ERC20_avg_time_between_contract_tnx': 0.0,
            'ERC20_min_val_rec': 0.0,
            'ERC20_max_val_rec': 0.0,
            'ERC20_avg_val_rec': 0.0,
            'ERC20_min_val_sent': 0.0,
            'ERC20_max_val_sent': 0.0,
            'ERC20_avg_val_sent': 0.0,
            'ERC20_min_val_sent_contract': 0.0,
            'ERC20_max_val_sent_contract': 0.0,
            'ERC20_avg_val_sent_contract': 0.0,
            'ERC20_uniq_sent_token_name': 0.0,
            'ERC20_uniq_rec_token_name': 0.0,
            'ERC20_most_sent_token_type': '',
            'ERC20_most_rec_token_type': ''
        }])

    # === All calculations (same as online version) ===
    time_diff = calc_time_diff(txs)
    sent_count, rec_count = calc_sent_received_counts(txs, address)
    avg_sent_between = calc_avg_min_between([tx for tx in txs if tx['from'].lower() == addr_lower])
    avg_rec_between = calc_avg_min_between([tx for tx in txs if tx['to'] and tx['to'].lower() == addr_lower])
    created_count = calc_created_contracts(txs, address)
    unique_rec_from, unique_sent_to = calc_unique_addresses(txs, address)

    (min_rec, max_rec, avg_rec, min_sent, max_sent, avg_sent,
     total_sent, total_rec) = calc_eth_values(txs, address)

    (min_sent_contract, max_sent_contract, avg_sent_contract,
     total_sent_contracts) = calc_eth_contract_values(txs, address, contract_cache)

    total_tx = calc_total_transactions(sent_count, rec_count)
    total_balance = calc_total_balance(total_sent, total_rec)
    total_erc_tnxs = calc_erc20_counts(erc_txs, address)

    (erc_total_rec, erc_total_sent, erc_min_rec, erc_max_rec, erc_avg_rec,
     erc_min_sent, erc_max_sent, erc_avg_sent) = calc_erc20_values(erc_txs, address)

    (erc_total_sent_contract, erc_min_sent_contract, erc_max_sent_contract,
     erc_avg_sent_contract) = calc_erc20_contract_values(erc_txs, address, contract_cache)

    (erc_uniq_sent_addr, erc_uniq_rec_addr,
     erc_uniq_sent_addr1, erc_uniq_rec_contract) = calc_erc20_unique_addresses(erc_txs, address)

    (erc_avg_sent_time, erc_avg_rec_time, erc_avg_rec2_time,
     erc_avg_contract_time) = calc_erc20_time_metrics(erc_txs, address, contract_cache)

    erc_uniq_sent_token, erc_uniq_rec_token = calc_erc20_token_names(erc_txs, address)
    most_sent_type, most_rec_type = calc_erc20_most_token_types(erc_txs, address)

    # Build final DataFrame (exact same structure)
    df = pd.DataFrame([{
        'Unnamed: 0': 0,
        'Index': 1,
        'Address': address,
        'FLAG': 2,
        'Contract': tempContract,
        'Avg_min_between_sent_tnx': avg_sent_between,
        'Avg_min_between_received_tnx': avg_rec_between,
        'Time_Diff_between_first_and_last_(Mins)': time_diff,
        'Sent_tnx': sent_count,
        'Received_Tnx': rec_count,
        'Number_of_Created_Contracts': created_count + tempContract,
        'Unique_Received_From_Addresses': unique_rec_from,
        'Unique_Sent_To_Addresses': unique_sent_to,
        'min_value_received': min_rec,
        'max_value_received': max_rec,
        'avg_val_received': avg_rec,
        'min_val_sent': min_sent,
        'max_val_sent': max_sent,
        'avg_val_sent': avg_sent,
        'min_value_sent_to_contract': min_sent_contract,
        'max_val_sent_to_contract': max_sent_contract,
        'avg_value_sent_to_contract': avg_sent_contract,
        'total_transactions_(including_tnx_to_create_contract': total_tx + tempContract,
        'total_Ether_sent': total_sent,
        'total_ether_received': total_rec,
        'total_ether_sent_contracts': total_sent_contracts,
        'total_ether_balance': total_balance,
        'Total_ERC20_tnxs': total_erc_tnxs,
        'ERC20_total_Ether_received': erc_total_rec,
        'ERC20_total_ether_sent': erc_total_sent,
        'ERC20_total_Ether_sent_contract': erc_total_sent_contract,
        'ERC20_uniq_sent_addr': float(erc_uniq_sent_addr),
        'ERC20_uniq_rec_addr': float(erc_uniq_rec_addr),
        'ERC20_uniq_sent_addr.1': float(erc_uniq_sent_addr1),
        'ERC20_uniq_rec_contract_addr': float(erc_uniq_rec_contract),
        'ERC20_avg_time_between_sent_tnx': erc_avg_sent_time,
        'ERC20_avg_time_between_rec_tnx': erc_avg_rec_time,
        'ERC20_avg_time_between_rec_2_tnx': erc_avg_rec2_time,
        'ERC20_avg_time_between_contract_tnx': erc_avg_contract_time,
        'ERC20_min_val_rec': erc_min_rec,
        'ERC20_max_val_rec': erc_max_rec,
        'ERC20_avg_val_rec': erc_avg_rec,
        'ERC20_min_val_sent': erc_min_sent,
        'ERC20_max_val_sent': erc_max_sent,
        'ERC20_avg_val_sent': erc_avg_sent,
        'ERC20_min_val_sent_contract': erc_min_sent_contract,
        'ERC20_max_val_sent_contract': erc_max_sent_contract,
        'ERC20_avg_val_sent_contract': erc_avg_sent_contract,
        'ERC20_uniq_sent_token_name': erc_uniq_sent_token,
        'ERC20_uniq_rec_token_name': erc_uniq_rec_token,
        'ERC20_most_sent_token_type': most_sent_type,
        'ERC20_most_rec_token_type': most_rec_type
    }])

    df.replace([np.inf, -np.inf], 0, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=-1e18, upper=1e18)

    print(f"Offline processing complete for {address}")
    return df

def load_real_data(csv_path):
    """
    Load single-row Etherscan data and clean it to match training pipeline
    """
    df = pd.read_csv(csv_path)
    print(f"Raw data loaded with shape: {df.shape}")

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    print(f"Cleaned column names: {len(df.columns)} columns")

    # Drop metadata and problematic columns (including any Unnamed:.* columns)
    columns_to_drop = [
        'Unnamed:_0', 'Index', 'Address',  # Metadata
        'ERC20_most_sent_token_type',  # Categorical 1
        'ERC20_most_rec_token_type',   # Categorical 2
    ]
    # Add any columns matching 'Unnamed:.*' (e.g., Unnamed:_0.1)
    columns_to_drop.extend([col for col in df.columns if re.match(r'Unnamed:.*', col)])
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    print(f"Dropping columns: {columns_to_drop}")

    df_clean = df.drop(columns=columns_to_drop, errors='ignore')

    # Fill NaNs with 0 (same as training)
    df_clean = df_clean.fillna(0)

    # NEW: Handle infinite values with 0
    df_clean.replace([np.inf, -np.inf], 0, inplace=True)

    # NEW: Clip extremely large values to prevent overflow
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].clip(lower=-1e18, upper=1e18)

    # Verify no infinite or excessively large values
    if np.isinf(df_clean[numeric_cols].values).any():
        print("WARNING: Infinite values still present!")
    elif (df_clean[numeric_cols].abs() > 1e18).any().any():
        print("WARNING: Excessively large values still present!")
    else:
        print("✓ No infinite or excessively large values in data.")

    # Ensure all numeric (force float64)
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

    print(f"Final cleaned data shape: {df_clean.shape}")
    print(f"All numeric: {df_clean.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()}")

    return df_clean

def shap_explainer(model, df_features):
    """
    Use SHAP to explain the model's prediction for the given features.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df_features)

    # For binary classification (fraud class)
    shap_df = pd.DataFrame({
        'feature': df_features.columns,
        'shap_value': shap_values.values[0, :]
    })
    shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
    shap_df = shap_df.sort_values('abs_shap', ascending=False).head(10)

    print(f"\n{'=' * 50}")
    print("SHAP EXPLANATION")
    print(f"{'=' * 50}")
    print("\nTop 10 Contributing Features to Fraud Probability:")
    print(shap_df[['feature', 'shap_value']].round(4).to_string(index=False))
    print("\n(Positive SHAP values increase the probability of Fraud, negative decrease it.)")

    # Save force plot
    shap.plots.force(shap_values[0], matplotlib=True)
    # plt.savefig('shap_force.png', bbox_inches='tight')
    # print("SHAP force plot saved to 'shap_force.png'")

def predict_fraud(input_file='out.csv', model_path='fraud_model.pkl', address=None):
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Run the pickle dump in your notebook first!")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)

    # Load the full data (including Address, FLAG)
    try:
        df = pd.read_csv(input_file)
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]

        if address:
            # Filter for the matching address
            df = df[df['Address'].str.lower() == address.lower()]
            if len(df) == 0:
                print(f"Error: Address {address} not found in {input_file}")
                sys.exit(1)
            if len(df) > 1:
                print(f"Warning: Multiple rows found for address {address}. Using the first one.")
                df = df.iloc[[0]]

        # Save the filtered DataFrame to a temporary CSV to pass to load_real_data
        temp_file = 'temp_filtered.csv'
        df.to_csv(temp_file, index=False)

        # Get cleaned features for prediction using the filtered data
        df_clean = load_real_data(temp_file)
        os.remove(temp_file)  # Clean up temporary file

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    row_data = df.iloc[0]
    addr = row_data.get('Address', 'Unknown')

    print(f"\n{'=' * 50}")
    print(f"PREDICTING FOR ETHERSCAN DATA")
    print(f"{'=' * 50}")
    print(f"Address: {addr}")
    print(f"Contract: {'Yes' if row_data.get('Contract', 0) == 1 else 'No'}")
    print(f"Total Transactions: {row_data.get('total_transactions_(including_tnx_to_create_contract', 0)}")
    print(f"Total Ether Sent: {row_data.get('total_Ether_sent', 0):.6f}")
    print(f"ERC20 Transactions: {row_data.get('Total_ERC20_tnxs', 0)}")
    print(f"{'=' * 50}")

    # Load the trained model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Define the expected 46 features (based on training data)
    expected_features = [
        'Contract', 'Avg_min_between_sent_tnx', 'Avg_min_between_received_tnx',
        'Time_Diff_between_first_and_last_(Mins)', 'Sent_tnx', 'Received_Tnx',
        'Number_of_Created_Contracts', 'Unique_Received_From_Addresses', 'Unique_Sent_To_Addresses',
        'min_value_received', 'max_value_received', 'avg_val_received', 'min_val_sent',
        'max_val_sent', 'avg_val_sent', 'min_value_sent_to_contract', 'max_val_sent_to_contract',
        'avg_value_sent_to_contract', 'total_transactions_(including_tnx_to_create_contract',
        'total_Ether_sent', 'total_ether_received', 'total_ether_sent_contracts',
        'total_ether_balance', 'Total_ERC20_tnxs', 'ERC20_total_Ether_received',
        'ERC20_total_ether_sent', 'ERC20_total_Ether_sent_contract', 'ERC20_uniq_sent_addr',
        'ERC20_uniq_rec_addr', 'ERC20_uniq_sent_addr.1', 'ERC20_uniq_rec_contract_addr',
        'ERC20_avg_time_between_sent_tnx', 'ERC20_avg_time_between_rec_tnx',
        'ERC20_avg_time_between_rec_2_tnx', 'ERC20_avg_time_between_contract_tnx',
        'ERC20_min_val_rec', 'ERC20_max_val_rec', 'ERC20_avg_val_rec',
        'ERC20_min_val_sent', 'ERC20_max_val_sent', 'ERC20_avg_val_sent',
        'ERC20_min_val_sent_contract', 'ERC20_max_val_sent_contract', 'ERC20_avg_val_sent_contract',
        'ERC20_uniq_sent_token_name', 'ERC20_uniq_rec_token_name'
    ]

    # Select only the expected features
    missing_features = [f for f in expected_features if f not in df_clean.columns]
    if missing_features:
        print(f"Error: Missing required features: {missing_features}")
        sys.exit(1)

    features = df_clean[expected_features]

    # Ensure we have exactly 46 features
    if len(features.columns) != 46:
        print(f"Error: Expected 46 features, got {len(features.columns)}")
        print("Available features:", list(features.columns))
        print("Missing features:", [f for f in expected_features if f not in features.columns])
        print("Extra features:", [f for f in features.columns if f not in expected_features])
        sys.exit(1)

    # Convert to float and reshape for prediction
    features_array = features.astype(float).values.reshape(1, -1)

    # Make prediction
    prediction = model.predict(features_array)[0]
    prob = model.predict_proba(features_array)[0]

    # Update FLAG in the full DF for the matching address
    full_df = pd.read_csv(input_file)
    full_df.columns = full_df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
    full_df.columns = [col.strip().replace(' ', '_') for col in full_df.columns]
    full_df.loc[full_df['Address'].str.lower() == addr.lower(), 'FLAG'] = prediction

    # Save back to the same file
    save_to_csv(full_df, input_file, mode='w')

    # Format output
    fraud_prob = prob[1] * 100
    non_fraud_prob = prob[0] * 100

    print(f"\nPREDICTION RESULTS")
    print(f"{'-' * 30}")
    print(f"Prediction: {'FRAUD (1)' if prediction == 1 else 'NON-FRAUD (0)'}")
    print(f"Confidence: {max(fraud_prob, non_fraud_prob):.1f}%")
    print(f"Fraud Probability: {fraud_prob:.2f}%")
    print(f"Non-Fraud Probability: {non_fraud_prob:.2f}%")

    if prediction == 1:
        print(f"\nALERT: This address shows fraudulent patterns!")
        print(f"   • High fraud probability: {fraud_prob:.1f}%")
        if fraud_prob > 95:
            print(f"   • CRITICAL: Extremely high confidence ({fraud_prob:.1f}%)")
        elif fraud_prob > 80:
            print(f"   • HIGH: Strong fraud indicators ({fraud_prob:.1f}%)")
    else:
        print(f"\nSAFE: This address appears legitimate")
        print(f"   • Low fraud probability: {fraud_prob:.1f}%")
        if fraud_prob > 10:
            print(f"   • MONITOR: Some suspicious activity detected ({fraud_prob:.1f}%)")

    print(f"\nRaw Probabilities: Non-Fraud={prob[0]:.4f}, Fraud={prob[1]:.4f}")
    print(f"{'=' * 50}")

    # Call SHAP explainer
    shap_explainer(model, features)


def main():
    parser = argparse.ArgumentParser(description="Ethereum Address Crawler and Fraud Predictor")
    parser.add_argument('--C', action='store_true', help='Crawl mode: Crawl data for the addresses')
    parser.add_argument('--CaP', action='store_true', help='Crawl and Predict mode: Crawl data and then predict fraud')
    parser.add_argument('--CO', action='store_true', help='Crawl Offline: Process raw/ data without API')
    parser.add_argument('--COaP', action='store_true', help='Crawl Offline and Predict')
    parser.add_argument('--P', nargs='?', const='out.csv',
                        help='Predict mode: Predict fraud from CSV file (default: out.csv)')
    parser.add_argument('--address', help='Ethereum addresses to crawl or verify (comma-separated)')
    parser.add_argument('--api_keys', help='Etherscan API keys (comma-separated, default from env)')

    args = parser.parse_args()

    mode_count = sum([args.C, args.CaP, args.CO, args.COaP, args.P is not None])
    if mode_count != 1:
        parser.error("Exactly one mode must be specified: --C, --CaP, or --P")

    if args.CO or args.COaP:
        if not args.address:
            parser.error("--address is required for --CO or --COaP")
        addresses = [a.strip() for a in args.address.split(',') if a.strip()]
        output_file = 'out.csv'

        for i, addr in enumerate(addresses, 1):
            print(f"\n[OFFLINE] Processing {i}/{len(addresses)}: {addr}")
            df_new = process_offline_address(addr)
            if df_new is not None:
                save_to_csv(df_new, output_file, mode='a')
                print(f"Saved offline data to {output_file}")
                if args.COaP:
                    predict_fraud(output_file, address=addr)
            else:
                print(f"Failed to process {addr} — missing raw files")

    if args.C or args.CaP:
        if not args.address:
            parser.error("--address is required for --C or --CaP modes")
        if not args.api_keys and not os.getenv('ETHERSCAN_API_KEY'):
            parser.error(
                "--api_keys is required (or set ETHERSCAN_API_KEY env with comma-separated keys) for --C or --CaP modes")

        # Split comma-separated addresses
        addresses = [addr.strip() for addr in args.address.split(',') if addr.strip()]
        if not addresses:
            parser.error("No valid addresses provided")

        crawl_output = 'out.csv'
        for i, addr in enumerate(addresses, 1):
            print(f"\nProcessing address {i}/{len(addresses)}: {addr}")
            # Perform crawl
            df_new = get_transaction_and_metadata(addr, args.api_keys)
            print(f"Transformed dataset row for {addr}:")
            print(df_new.to_string(index=False))
            save_to_csv(df_new, crawl_output, mode='a')  # Append mode
            print(f"Data saved to '{crawl_output}' with FLAG=2 (crawled, not predicted)")

            if args.CaP:
                # Proceed to predict using the crawled data
                predict_fraud(crawl_output, address=addr)

    elif args.P is not None:
        if not args.address:
            parser.error("--address is required for --P mode to verify the data")
        # Split comma-separated addresses
        addresses = [addr.strip() for addr in args.address.split(',') if addr.strip()]
        if not addresses:
            parser.error("No valid addresses provided")

        input_file = args.P
        for i, addr in enumerate(addresses, 1):
            print(f"\nProcessing address {i}/{len(addresses)}: {addr}")
            predict_fraud(input_file, address=addr)

if __name__ == "__main__":
    main()