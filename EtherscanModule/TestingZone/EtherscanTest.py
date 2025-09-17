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


def get_transactions(address, action='txlist'):
    """
    Fetch transaction history (normal, token, or internal) for a given Ethereum address
    """
    params = {
        'module': 'account',
        'action': action,
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'sort': 'desc',
        'apikey': ETHERSCAN_API_KEY
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if data['status'] == '1' and data['message'] == 'OK':
            return data['result']
        else:
            print(f"API Error for {action}: {data['message']}")
            return []
    except requests.RequestException as e:
        print(f"Error fetching {action}: {e}")
        return []


def transactions_to_csv(normal_txs, token_txs, internal_txs, address):
    if not (normal_txs or token_txs or internal_txs):
        print("No transactions to save")
        return None

    data = []

    # Normal ETH transactions
    for tx in normal_txs:
        if tx['isError'] == '0':
            stat = 'Success'
        else:
            stat = 'Failed',
        data.append({
            'TxType': 'ETH',
            'Timestamp': datetime.fromtimestamp(int(tx['timeStamp'])).strftime('%Y-%m-%d %H:%M:%S'),
            'TxHash': tx['hash'],
            'Block': tx['blockNumber'],
            'From': tx['from'],
            'To': tx['to'] if tx['to'] else '',
            'Value': float(tx['value']) / 1e18,
            'Gas': tx['gas'],
            'GasPrice': float(tx['gasPrice']) / 1e9,
            'Status': str(stat),
            'TokenName': '',
            'TokenSymbol': '',
            'TokenDecimal': '',
            'ContractAddress': tx.get('contractAddress', ''),
            'InternalType': ''
        })
    for tx in token_txs:
        data.append({
            'TxType': 'ERC20',
            'Timestamp': datetime.fromtimestamp(int(tx['timeStamp'])).strftime('%Y-%m-%d %H:%M:%S'),
            'TxHash': tx['hash'],
            'Block': tx['blockNumber'],
            'From': tx['from'],
            'To': tx['to'],
            'Value': float(tx['value']) / (10 ** int(tx['tokenDecimal'])),
            'Gas': tx['gas'],
            'GasPrice': float(tx['gasPrice']) / 1e9,
            'Status': stat,
            'TokenName': tx['tokenName'],
            'TokenSymbol': tx['tokenSymbol'],
            'TokenDecimal': tx['tokenDecimal'],
            'ContractAddress': tx['contractAddress'],
            'InternalType': ''
        })

    # Internal transactions (e.g., contract creations)
    for tx in internal_txs:
        if tx['isError'] == '0':
            stat = 'Success'
        else:
            stat = 'Failed',
        data.append({
            'TxType': 'Internal',
            'Timestamp': datetime.fromtimestamp(int(tx['timeStamp'])).strftime('%Y-%m-%d %H:%M:%S'),
            'TxHash': tx['hash'],
            'Block': tx['blockNumber'],
            'From': tx['from'],
            'To': tx['to'] if tx['to'] else '',
            'Value': float(tx['value']) / 1e18,
            'Gas': '',
            'GasPrice': '',
            'Status': stat,
            'TokenName': '',
            'TokenSymbol': '',
            'TokenDecimal': '',
            'ContractAddress': tx.get('contractAddress', ''),
            'InternalType': tx['type']
        })

    df = pd.DataFrame(data)
    filename = f"transactions_{address[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"Transaction history saved to {filename}")
    return df


def token_transactions_to_metadata(address, normal_txs, token_txs, internal_txs):
    """
    Extract single-value metadata for anomaly detection dataset
    """
    metadata = {'Address': address}

    # Normal transactions
    sent_txs = [tx for tx in normal_txs if tx['from'].lower() == address.lower()]
    received_txs = [tx for tx in normal_txs if tx['to'].lower() == address.lower()]

    # ERC20 token transactions
    erc20_sent = [tx for tx in token_txs if tx['from'].lower() == address.lower()]
    erc20_received = [tx for tx in token_txs if tx['to'].lower() == address.lower()]

    # Contract creations
    created_contracts = [tx for tx in internal_txs if tx['from'].lower() == address.lower() and tx['type'] == 'create']

    # Single-value aggregates
    metadata['Sent tnx'] = len(sent_txs)
    metadata['Received Tnx'] = len(received_txs)
    metadata['Number of Created Contracts'] = len(created_contracts)
    metadata['Unique Received From Addresses'] = len(set(tx['from'].lower() for tx in received_txs))
    metadata['Unique Sent To Addresses'] = len(set(tx['to'].lower() for tx in sent_txs if tx['to']))

    metadata['min value received'] = min((float(tx['value']) / 1e18 for tx in received_txs), default=0)
    metadata['max value received'] = max((float(tx['value']) / 1e18 for tx in received_txs), default=0)
    metadata['avg val received'] = sum(float(tx['value']) / 1e18 for tx in received_txs) / len(
        received_txs) if received_txs else 0
    metadata['min val sent'] = min((float(tx['value']) / 1e18 for tx in sent_txs), default=0)
    metadata['max val sent'] = max((float(tx['value']) / 1e18 for tx in sent_txs), default=0)
    metadata['avg val sent'] = sum(float(tx['value']) / 1e18 for tx in sent_txs) / len(sent_txs) if sent_txs else 0

    contract_txs = [tx for tx in sent_txs if tx.get('contractAddress')]
    metadata['min value sent to contract'] = min((float(tx['value']) / 1e18 for tx in contract_txs), default=0)
    metadata['max val sent to contract'] = max((float(tx['value']) / 1e18 for tx in contract_txs), default=0)
    metadata['avg value sent to contract'] = sum(float(tx['value']) / 1e18 for tx in contract_txs) / len(
        contract_txs) if contract_txs else 0

    metadata['total transactions (including tnx to create contract)'] = len(normal_txs) + len(created_contracts)
    metadata['total Ether sent'] = sum(float(tx['value']) / 1e18 for tx in sent_txs)
    metadata['total ether received'] = sum(float(tx['value']) / 1e18 for tx in received_txs)
    metadata['total ether sent contracts'] = sum(float(tx['value']) / 1e18 for tx in contract_txs)
    metadata['total ether balance'] = metadata['total ether received'] - metadata['total Ether sent']

    metadata['Total ERC20 tnxs'] = len(token_txs)
    metadata['ERC20 total Ether received'] = sum(
        float(tx['value']) / (10 ** int(tx['tokenDecimal'])) for tx in erc20_received)
    metadata['ERC20 total ether sent'] = sum(float(tx['value']) / (10 ** int(tx['tokenDecimal'])) for tx in erc20_sent)
    metadata['ERC20 total Ether sent contract'] = sum(
        float(tx['value']) / (10 ** int(tx['tokenDecimal'])) for tx in erc20_sent if tx.get('contractAddress')) / len(
        [tx for tx in erc20_sent if tx.get('contractAddress')]) if any(
        tx.get('contractAddress') for tx in erc20_sent) else 0
    metadata['ERC20 uniq sent addr'] = len(set(tx['to'].lower() for tx in erc20_sent if tx['to']))
    metadata['ERC20 uniq rec addr'] = len(set(tx['from'].lower() for tx in erc20_received))
    metadata['ERC20 uniq sent addr.1'] = metadata['ERC20 uniq sent addr']
    metadata['ERC20 uniq rec contract addr'] = len(set(tx['contractAddress'].lower() for tx in erc20_received))

    metadata['ERC20 min val rec'] = min((float(tx['value']) / (10 ** int(tx['tokenDecimal'])) for tx in erc20_received),
                                        default=0)
    metadata['ERC20 max val rec'] = max((float(tx['value']) / (10 ** int(tx['tokenDecimal'])) for tx in erc20_received),
                                        default=0)
    metadata['ERC20 avg val rec'] = sum(
        float(tx['value']) / (10 ** int(tx['tokenDecimal'])) for tx in erc20_received) / len(
        erc20_received) if erc20_received else 0
    metadata['ERC20 min val sent'] = min((float(tx['value']) / (10 ** int(tx['tokenDecimal'])) for tx in erc20_sent),
                                         default=0)
    metadata['ERC20 max val sent'] = max((float(tx['value']) / (10 ** int(tx['tokenDecimal'])) for tx in erc20_sent),
                                         default=0)
    metadata['ERC20 avg val sent'] = sum(
        float(tx['value']) / (10 ** int(tx['tokenDecimal'])) for tx in erc20_sent) / len(
        erc20_sent) if erc20_sent else 0
    metadata['ERC20 min val sent contract'] = min(
        (float(tx['value']) / (10 ** int(tx['tokenDecimal'])) for tx in erc20_sent if tx.get('contractAddress')),
        default=0)
    metadata['ERC20 max val sent contract'] = max(
        (float(tx['value']) / (10 ** int(tx['tokenDecimal'])) for tx in erc20_sent if tx.get('contractAddress')),
        default=0)
    metadata['ERC20 avg val sent contract'] = sum(
        float(tx['value']) / (10 ** int(tx['tokenDecimal'])) for tx in erc20_sent if tx.get('contractAddress')) / len(
        [tx for tx in erc20_sent if tx.get('contractAddress')]) if any(
        tx.get('contractAddress') for tx in erc20_sent) else 0

    metadata['ERC20 uniq sent token name'] = len(set(tx['tokenName'] for tx in erc20_sent))
    metadata['ERC20 uniq rec token name'] = len(set(tx['tokenName'] for tx in erc20_received))
    sent_token_counts = pd.Series([tx['tokenName'] for tx in erc20_sent]).value_counts()
    metadata['ERC20 most sent token type'] = sent_token_counts.index[0] if not sent_token_counts.empty else ''
    rec_token_counts = pd.Series([tx['tokenName'] for tx in erc20_received]).value_counts()
    metadata['ERC20_most_rec_token_type'] = rec_token_counts.index[0] if not rec_token_counts.empty else ''

    df = pd.DataFrame([metadata])
    filename = f"metadata_{address[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"Metadata saved to {filename}")


def main():
    address = "0x007cafba4edd0e25a664530e3cac0ffadb101fd8"

    if not (len(address) == 42 and address.startswith('0x')):
        print("Invalid Ethereum address format")
        return

    normal_txs = get_transactions(address, 'txlist')
    token_txs = get_transactions(address, 'tokentx')
    internal_txs = get_transactions(address, 'txlistinternal')

    transactions_to_csv(normal_txs, token_txs, internal_txs, address)
    token_transactions_to_metadata(address, normal_txs, token_txs, internal_txs)


if __name__ == "__main__":
    if not ETHERSCAN_API_KEY:
        print("Error: Etherscan API key not found in .env file")
    else:
        main()