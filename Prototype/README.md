# Ethereum Fraud Detector

CLI tool to crawl Ethereum addresses and detect fraud using XGBoost.

## Quick Start

### Setup
```
pip install requests pandas numpy xgboost python-dotenv
echo "ETHERSCAN_API_KEY=YourApiKey" > .env
```

### Usage 

    usage: main.py [-h] [--C] [--CaP] [--P [P]] [--address ADDRESS] [--api_key API_KEY]

    Ethereum Address Crawler and Fraud Predictor

    options:
    -h, --help         show this help message and exit
    --C                Crawl mode: Crawl data for the address
    --CO               Offline Calculate mode: Recalculate data row with crawled data in /raw
    --COaP             Offline Calculate and Predict mode: Recalculate data row and predict with crawled data in /raw
    --CaP              Crawl and Predict mode: Crawl data and then predict fraud
    --P [P]            Predict mode: Predict fraud from CSV file (default: out.csv)
    --address ADDRESS  Ethereum address to crawl or verify
    --api_key API_KEY  Etherscan API key (default from env)

### Example

#### Crawl Address Data
```
python3 main.py --C --address 0x001b28141562bc2601694d27c3f5fda2c06c234c,0x0045eb75acf6e1cb1d9ec41e352a879e2cd50b35 --api_keys YourApiKey1,YourApiKey2
```
Output: out.csv with 47 behavioral features

#### Predict Fraud (from CSV)
```
python main.py --P --address 0x001b28141562bc2601694d27c3f5fda2c06c234c           # Uses out.csv
python main.py --P out.csv --address 0x001b28141562bc2601694d27c3f5fda2c06c234c   # Uses specific file
```
Output: Fraud probability & risk level

#### Full Pipeline (Crawl + Predict)
```
python main.py --CaP --address 0x001b28141562bc2601694d27c3f5fda2c06c234c,0x0045eb75acf6e1cb1d9ec41e352a879e2cd50b35 --api_keys YourApiKey1,YourApiKey2
```
Example Output: Features + "🔔 FRAUD (98.7%)" or "✅ SAFE"

### Note

Output files after running the script:

```
raw/contracts-<address>.json          # All addresses involved with the target, indicating contract status (true/false).
raw/erc20_transactions-<address>.csv  # Raw ERC20 transaction history for the address.
raw/transactions-<address>.csv        # Raw transaction history for the address.
raw/contracts-total.json             # Global cache of contract statuses for all crawled addresses (true/false).
out.csv                              # Processed behavioral features for one or more addresses, used for prediction.
```

- **Multiple Addresses**: The `--address` argument accepts a comma-separated list of Ethereum addresses (e.g., `--address addr1,addr2`). Each address is processed sequentially, appending results to `out.csv`.
- **Multiple API Keys**: The `--api_keys` argument accepts a comma-separated list of Etherscan API keys (e.g., `--api_keys key1,key2`). Alternatively, set `ETHERSCAN_API_KEY=key1,key2` in `.env`. More keys speed up contract status crawling by distributing requests in parallel, respecting Etherscan's 5 calls/second per key limit.
- **Optimization Tip**: Crawling high-interaction addresses (e.g., `0x0000000000000000000000000000000000000000`) populates `contracts-total.json` with many contract statuses, accelerating future crawls for other addresses.

#### Reading the Output CSV
The `out.csv` file contains processed data for Ethereum addresses, with each row representing one address and its associated behavioral features. It is designed to store data from multiple addresses (appended during `--C` or `--CaP` modes) and is used as input for fraud prediction in `--P` mode.

**Structure of `out.csv`:**
- **Columns**: 49 columns, including:
  - `Address`: The Ethereum address (e.g., `0x001b28141562bc2601694d27c3f5fda2c06c234c`).
  - `FLAG`: Status of the address:
    - `2`: Data crawled but not yet predicted (after `--C` or initial `--CaP`).
    - `1`: Predicted as fraudulent by the model.
    - `0`: Predicted as safe (non-fraudulent).
  - 46 numerical features (e.g., `Avg_min_between_sent_tnx`, `total_Ether_sent`, `Total_ERC20_tnxs`) used by the XGBoost model for prediction.
  - Two categorical features: `ERC20_most_sent_token_type` and `ERC20_most_rec_token_type` (not used in prediction but included for analysis).
- **Rows**: Each row corresponds to one Ethereum address. Running `--C` or `--CaP` with a new address appends a new row to `out.csv`, preserving existing data.

**How to Read `out.csv`:**
1. **Check the `FLAG` Column**:
   - A `FLAG` of `2` indicates the address was recently crawled but hasn’t been analyzed for fraud. Run `--P out.csv --address <address>` to predict and update the `FLAG` to `0` (safe) or `1` (fraud).
   - A `FLAG` of `1` means the address was predicted as fraudulent, with a high fraud probability (check console output for details).
   - A `FLAG` of `0` means the address was predicted as safe.
2. **Analyze Features**:
   - Columns like `total_Ether_sent`, `Total_ERC20_tnxs`, and `Unique_Received_From_Addresses` provide insights into the address’s transaction behavior.
   - Use these to understand why an address might be flagged as fraudulent (e.g., high `total_Ether_sent` with few `Unique_Sent_To_Addresses` might indicate suspicious activity).
3. **Manual Editing for Testing**:
   - You can edit feature values in `out.csv` (e.g., increase `total_Ether_sent`) to test how changes affect fraud predictions.
   - Save the edited file and run `--P out.csv --address <address>` to re-predict and update the `FLAG`.
   - Ensure the `Address` column matches the `--address` argument when re-running predictions.
4. **Viewing Raw Data**:
   - For detailed transaction data, check the address-specific files in the `raw/` directory (e.g., `raw/transactions-<address>.csv`).
   - These provide the raw transaction and ERC20 data used to compute the features in `out.csv`.

**Example**:
```csv
Address,FLAG,Contract,...,total_Ether_sent,Total_ERC20_tnxs,...
0x001b28141562bc2601694d27c3f5fda2c06c234c,1,0,...,100.5,50,...
0x1234567890abcdef1234567890abcdef12345678,2,1,...,10.2,20,...
```
- The first address has `FLAG=1` (fraudulent), indicating a high fraud probability from a previous `--CaP` or `--P` run.
- The second address has `FLAG=2` (crawled, not predicted), ready for `--P` analysis.

**Tips**:
- Use a CSV viewer (e.g., Excel, pandas in Python) to inspect `out.csv`.
- If `out.csv` grows large, filter rows by `Address` or `FLAG` to focus on specific addresses or statuses.
- Avoid manually adding index columns (e.g., `Unnamed:_0`) to prevent errors during prediction.