### Usage

    usage: EtherscanAPI.py [-h] --address ADDRESS [--api_key API_KEY]

    options:
    -h, --help         show this help message and exit
    --address ADDRESS
    --api_key API_KEY

In case you don't want to type API_KEY every run. You can store it in .env file next to .py file
`ETHERSCAN_API_KEY=YourEtherscanAPI`

### Output structure

    out.csv     #This file will be use for prediction with ML model later (The predict module is underway)
    transactions.csv        #[raw data] Transaction history of address, 
    erc20_transactions.csv  #[raw data] Erc20 Transaction history of 
    contracts.json          #[raw data] Involved address in transaction history