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
    --CaP              Crawl and Predict mode: Crawl data and then predict fraud
    --P [P]            Predict mode: Predict fraud from CSV file (default: input.csv)
    --address ADDRESS  Ethereum address to crawl
    --api_key API_KEY  Etherscan API key (default from env)

### Example

#### Crawl Address Data
```
python3 main.py --C --address 0x001b28141562bc2601694d27c3f5fda2c06c234c --api_key YourEtherscanAPI
```
Output: out.csv with 47 behavioral features

#### Predict Fraud (from CSV)
```
python predict.py --P           # Uses input.csv
python predict.py --P out.csv   # Uses specific file
```
Output: Fraud probability & risk level

#### Full Pipeline (Crawl + Predict)
```
bashpython predict.py --CaP --address 0x001b28141562bc2601694d27c3f5fda2c06c234c --api_key YourApiKey
```
Example Output: Features + "🔔 FRAUD (98.7%)" or "✅ SAFE"

### Note

Output after running python file:

```
contracts.json          #All address involved in this target, contract or not.
erc20_transactions.csv  #Raw ERC20 Transaction History.
transactions.csv        #Raw Transaction History.
out.csv                 #The output of crawl, file will be used to predict.
```