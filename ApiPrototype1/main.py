from flask import Flask, request, jsonify
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

app = Flask(__name__)

# Load environment variables
load_dotenv()
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')
FLASK_API_KEY = os.getenv('FLASK_API_KEY')

# Load out.csv into memory
OUT_CSV = 'out.csv'
EXPLAIN_JSON = 'explain.json'
global_df = pd.DataFrame()
if os.path.exists(OUT_CSV):
    try:
        global_df = pd.read_csv(OUT_CSV)
        global_df.columns = global_df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
        global_df.columns = [col.strip().replace(' ', '_') for col in global_df.columns]
    except Exception:
        global_df = pd.DataFrame()

# Etherscan API endpoint
BASE_URL = "https://api.etherscan.io/v2/api"

# Expected features for prediction
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

def validate_address(address):
    """Check if address is valid Ethereum format."""
    return bool(re.match(r'^0x[a-fA-F0-9]{40}$', address.lower()))

def validate_api_key(api_key):
    """Check if provided API key matches FLASK_API_KEY."""
    return api_key == FLASK_API_KEY

def is_contract(addr, api_key):
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
        resp = requests.get(BASE_URL, params=params)
        if resp.status_code != 200:
            return False
        data = resp.json()
        code = data.get('result')
        return len(code) > 2
    except Exception:
        return False

def get_txlist(address, api_key, action='txlist'):
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
        time.sleep(0.2)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if data.get('status') == '0':
            return []
        return data['result']
    except Exception:
        return []

def avg_min_erc(txs_list):
    if len(txs_list) < 2:
        return 0.0
    times = sorted(int(tx['timeStamp']) for tx in txs_list)
    diffs = [(times[i+1] - times[i]) / 60.0 for i in range(len(times)-1)]
    return sum(diffs) / len(diffs) if diffs else 0.0

def save_to_csv(df, output_path='out.csv', mode='w'):
    global global_df
    try:
        if mode == 'a' and os.path.exists(output_path):
            existing_df = global_df
            existing_df = existing_df[existing_df['Address'] != df['Address'].iloc[0]]
            df = pd.concat([existing_df, df], ignore_index=True)
        global_df = df
        # Write atomically to avoid corruption
        temp_file = output_path + '.tmp'
        df.to_csv(temp_file, index=False)
        os.replace(temp_file, output_path)
        return True
    except Exception:
        return False

def wei_to_eth(v_str):
    return int(v_str) / 1e18 if v_str else 0.0

def adjusted_value(tx):
    decimals = int(tx.get('tokenDecimal', '0'))
    return float(int(tx['value']) / (10 ** decimals)) if tx['value'] else 0.0

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

def save_raw_data(txs, erc_txs, contract_cache, address):
    try:
        os.makedirs('raw', exist_ok=True)
        pd.DataFrame(txs).to_csv(f'raw/transactions-{address.lower()}.csv', index=False)
        pd.DataFrame(erc_txs).to_csv(f'raw/erc20_transactions-{address.lower()}.csv', index=False)
        with open(f'raw/contracts-{address.lower()}.json', 'w') as f:
            json.dump(contract_cache, f)
        return True
    except Exception:
        return False

def get_transaction_and_metadata(address, api_key):
    txs = get_txlist(address, api_key, 'txlist')
    erc_txs = get_txlist(address, api_key, 'tokentx')

    tempContract = 1 if is_contract(address, api_key) else 0

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

    contract_cache = {addr: is_contract(addr, api_key) for addr in unique_addrs}

    # Save raw data
    save_raw_data(txs, erc_txs, contract_cache, address)

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
    sent_count, rec_count = calc_sent_received_counts(txs, address)
    avg_sent_between = calc_avg_min_between([tx for tx in txs if tx['from'].lower() == address.lower()])
    avg_rec_between = calc_avg_min_between([tx for tx in txs if tx['to'] and tx['to'].lower() == address.lower()])
    created_count = calc_created_contracts(txs, address)
    unique_rec_from, unique_sent_to = calc_unique_addresses(txs, address)
    min_rec, max_rec, avg_rec, min_sent, max_sent, avg_sent, total_sent, total_rec = calc_eth_values(txs, address)
    min_sent_contract, max_sent_contract, avg_sent_contract, total_sent_contracts = calc_eth_contract_values(txs, address, contract_cache)
    total_tx = calc_total_transactions(sent_count, rec_count)
    total_balance = calc_total_balance(total_sent, total_rec)
    total_erc_tnxs = calc_erc20_counts(erc_txs, address)
    erc_total_rec, erc_total_sent, erc_min_rec, erc_max_rec, erc_avg_rec, erc_min_sent, erc_max_sent, erc_avg_sent = calc_erc20_values(erc_txs, address)
    erc_total_sent_contract, erc_min_sent_contract, erc_max_sent_contract, erc_avg_sent_contract = calc_erc20_contract_values(erc_txs, address, contract_cache)
    erc_uniq_sent_addr, erc_uniq_rec_addr, erc_uniq_sent_addr1, erc_uniq_rec_contract = calc_erc20_unique_addresses(erc_txs, address)
    erc_avg_sent_time, erc_avg_rec_time, erc_avg_rec2_time, erc_avg_contract_time = calc_erc20_time_metrics(erc_txs, address, contract_cache)
    erc_uniq_sent_token, erc_uniq_rec_token = calc_erc20_token_names(erc_txs, address)
    most_sent_type, most_rec_type = calc_erc20_most_token_types(erc_txs, address)

    FirstContract = 0
    if is_contract(address, api_key):
        FirstContract = 1

    return pd.DataFrame([{
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
        'Number_of_Created_Contracts': created_count + FirstContract,
        'Unique_Received_From_Addresses': unique_rec_from,
        'Unique_Sent_To_Addresses': unique_sent_to,
        'min_value_received': min_rec,
        'max_value_received': max_rec,
        'avg_val_received': avg_rec,
        'min_val_sent': min_sent,
        'max_val_sent': max_sent,
        'avg_val_sent': avg_sent,
        'min_value_sent_to_contract': 0.0,  # min_sent_contract,
        'max_val_sent_to_contract': 0.0,  # max_sent_contract,
        'avg_value_sent_to_contract': 0.0,  # avg_sent_contract,
        'total_transactions_(including_tnx_to_create_contract': total_tx + FirstContract,
        'total_Ether_sent': total_sent,
        'total_ether_received': total_rec,
        'total_ether_sent_contracts': 0.0,  # total_sent_contracts,
        'total_ether_balance': total_balance,
        'Total_ERC20_tnxs': total_erc_tnxs,
        'ERC20_total_Ether_received': erc_total_rec,
        'ERC20_total_ether_sent': erc_total_sent,
        'ERC20_total_Ether_sent_contract': erc_total_sent_contract,
        'ERC20_uniq_sent_addr': float(erc_uniq_sent_addr),
        'ERC20_uniq_rec_addr': float(erc_uniq_rec_addr),
        'ERC20_uniq_sent_addr.1': float(erc_uniq_sent_addr1),
        'ERC20_uniq_rec_contract_addr': float(erc_uniq_rec_contract),
        'ERC20_avg_time_between_sent_tnx': 0.0,  # erc_avg_sent_time,
        'ERC20_avg_time_between_rec_tnx': 0.0,  # erc_avg_rec_time,
        'ERC20_avg_time_between_rec_2_tnx': 0.0,  # erc_avg_rec2_time,
        'ERC20_avg_time_between_contract_tnx': 0.0,  # erc_avg_contract_time,
        'ERC20_min_val_rec': erc_min_rec,
        'ERC20_max_val_rec': erc_max_rec,
        'ERC20_avg_val_rec': erc_avg_rec,
        'ERC20_min_val_sent': erc_min_sent,
        'ERC20_max_val_sent': erc_max_sent,
        'ERC20_avg_val_sent': erc_avg_sent,
        'ERC20_min_val_sent_contract': 0.0,  # erc_min_sent_contract,
        'ERC20_max_val_sent_contract': 0.0,  # erc_max_sent_contract,
        'ERC20_avg_val_sent_contract': 0.0,  # erc_avg_sent_contract,
        'ERC20_uniq_sent_token_name': erc_uniq_sent_token,
        'ERC20_uniq_rec_token_name': erc_uniq_rec_token,
        'ERC20_most_sent_token_type': most_sent_type,
        'ERC20_most_rec_token_type': most_rec_type
    }])

def load_real_data(df):
    columns_to_drop = [
        'Unnamed:_0', 'Index', 'Address',
        'ERC20_most_sent_token_type', 'ERC20_most_rec_token_type'
    ]
    columns_to_drop.extend([col for col in df.columns if re.match(r'Unnamed:.*', col)])
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_clean = df.drop(columns=columns_to_drop, errors='ignore')
    df_clean = df_clean.fillna(0)
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    return df_clean

def shap_explainer(model, df_features):
    # Use a simple background (zeros) for LinearExplainer
    background = np.zeros((1, len(expected_features)))
    background_df = pd.DataFrame(background, columns=expected_features)
    explainer = shap.LinearExplainer(model, background_df)
    shap_values = explainer(df_features)
    shap_df = pd.DataFrame({
        'feature': df_features.columns,
        'shap_value': shap_values.values[0, :]
    })
    return shap_df.to_dict(orient='records')

def save_shap_explanation(address, shap_values):
    try:
        explain_data = {}
        if os.path.exists(EXPLAIN_JSON):
            with open(EXPLAIN_JSON, 'r') as f:
                explain_data = json.load(f)
        explain_data[address.lower()] = shap_values
        with open(EXPLAIN_JSON + '.tmp', 'w') as f:
            json.dump(explain_data, f)
        os.replace(EXPLAIN_JSON + '.tmp', EXPLAIN_JSON)
        return True
    except Exception:
        return False

def predict_fraud(df, address, model_path='fraud_model.pkl'):
    try:
        df = df[df['Address'].str.lower() == address.lower()]
        if len(df) == 0:
            return None, None, None
        if len(df) > 1:
            df = df.iloc[[0]]
        temp_file = 'temp_filtered.csv'
        df.to_csv(temp_file, index=False)
        df_clean = load_real_data(df)
        os.remove(temp_file)
        row_data = df.iloc[0]
        if len(df_clean.columns) != 46:
            return None, None, None
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        features = df_clean[expected_features]
        features_array = features.astype(float).values.reshape(1, -1)
        prediction = model.predict(features_array)[0]
        shap_values = shap_explainer(model, features)
        global_df.loc[global_df['Address'].str.lower() == address.lower(), 'FLAG'] = prediction
        save_to_csv(global_df, OUT_CSV, mode='w')
        save_shap_explanation(address, shap_values)
        return row_data, prediction, shap_values
    except Exception:
        return None, None, None

@app.route('/<api_key>/crawl', methods=['GET'])
def crawl(api_key):
    if not validate_api_key(api_key):
        return jsonify({'Status': 1})
    address = request.args.get('address')
    if not address or not validate_address(address):
        return jsonify({'Status': 1})
    df_new = get_transaction_and_metadata(address, ETHERSCAN_API_KEY)
    if df_new is None:
        return jsonify({'Status': 1})
    if not save_to_csv(df_new, OUT_CSV, mode='a'):
        return jsonify({'Status': 1})
    return jsonify({
        'Status': 0,
        'Data': df_new.to_dict(orient='records')[0]
    })

@app.route('/<api_key>/crawlAndPredict', methods=['GET'])
def crawl_and_predict(api_key):
    if not validate_api_key(api_key):
        return jsonify({'Status': 1})
    address = request.args.get('address')
    if not address or not validate_address(address):
        return jsonify({'Status': 1})
    df_new = get_transaction_and_metadata(address, ETHERSCAN_API_KEY)
    if df_new is None:
        return jsonify({'Status': 1})
    if not save_to_csv(df_new, OUT_CSV, mode='a'):
        return jsonify({'Status': 1})
    row_data, prediction, shap_values = predict_fraud(global_df, address)
    if row_data is None:
        return jsonify({'Status': 1})
    return jsonify({
        'Status': 0,
        'Fraud': int(prediction),
        'Data': row_data.to_dict(),
        'Explain': shap_values
    })

@app.route('/<api_key>/Predict', methods=['GET'])
def predict(api_key):
    if not validate_api_key(api_key):
        return jsonify({'Status': 1})
    address = request.args.get('address')
    if not address or not validate_address(address):
        return jsonify({'Status': 1})
    row_data, prediction, shap_values = predict_fraud(global_df, address)
    if row_data is None:
        return jsonify({'Status': 1})
    return jsonify({
        'Status': 0,
        'Fraud': int(prediction),
        'Data': row_data.to_dict(),
        'Explain': shap_values
    })

@app.route('/<api_key>/getResult', methods=['GET'])
def get_result(api_key):
    if not validate_api_key(api_key):
        return jsonify({'Status': 1})
    address = request.args.get('address')
    if not address or not validate_address(address):
        return jsonify({'Status': 1})
    df = global_df[global_df['Address'].str.lower() == address.lower()]
    if len(df) == 0:
        return jsonify({'Status': 1})
    row_data = df.iloc[0]
    explain_data = []
    try:
        if os.path.exists(EXPLAIN_JSON):
            with open(EXPLAIN_JSON, 'r') as f:
                explain_data = json.load(f).get(address.lower(), [])
    except Exception:
        pass
    return jsonify({
        'Status': 0,
        'Fraud': int(row_data['FLAG']),
        'Data': row_data.to_dict(),
        'Explain': explain_data
    })

@app.route('/<api_key>/List', methods=['GET'])
def list_addresses(api_key):
    if not validate_api_key(api_key):
        return jsonify({'Status': 1})
    if global_df.empty:
        return jsonify({'Status': 1})
    addresses = global_df['Address'].tolist()
    return jsonify({
        'Status': 0,
        'Data': addresses
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)