#!/bin/bash
# Configuration for Spot markets
SPOT_ENABLED=false  # Set to false to skip spot downloads
SPOT_SYMBOLS=(BTCUSDT)

# Configuration for Futures markets
FUTURES_ENABLED=true  # Set to false to skip futures downloads
CM_OR_UM="um"  # "cm" or "um" for futures type
FUTURES_SYMBOLS=(BTCUSDT ETHUSDT BNBUSDT DOGEUSDT SHIBUSDT ADAUSDT XRPUSDT SOLUSDT LTCUSDT BCHUSDT MATICUSDT LINKUSDT AVAXUSDT DOTUSDT FTMUSDT UNIUSDT AAVEUSDT XLMUSDT TRXUSDT CROUSDT) #Add more symbols as needed

# Shared configurations
INTERVALS=("1m") #("1m" "5m" "15m" "30m" "1h" "2h" "4h" "6h" "8h" "12h" "1d" "3d" "1w" "1mo")
YEARS=("2020" "2021" "2022" "2023" "2024")
MONTHS=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12")

# Proxy settings
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

# Base directories
BASE_DIR="."
SPOT_DIR="${BASE_DIR}/spot"
UM_DIR="${BASE_DIR}/um"
CM_DIR="${BASE_DIR}/cm"
SYMBOL_TABLE_DIR="${BASE_DIR}/symbols"

# Data URLs
SPOT_BASE_URL="https://data.binance.vision/data/spot/monthly/klines"
FUTURES_BASE_URL="https://data.binance.vision/data/futures/${CM_OR_UM}/monthly/klines"

# Download all symbols from Spot, USD-M Futures and COIN-M Futures, into three separate files: `symbols.txt`, `um_symbols.txt` and `cm_symbols.txt`.
# Requires `jq` to be installed: https://stedolan.github.io/jq/

# +-------------+-------------+---------------------------+---------------+-------------+----------------+--------------------+
# | Market Type | Environment | REST API URI              | WS API URI    | REST Port   | WS Port        | SSL Verify Peer    |
# +-------------+-------------+---------------------------+---------------+-------------+----------------+--------------------+
# | USDM        | Live        | fapi.binance.com          | fstream.      | 443         | 443            | true               |
# |             |             |                           | binance.com   |             |                |                    |
# |             +-------------+---------------------------+---------------+-------------+----------------+--------------------+
# |             | Testnet     | testnet.binancefuture.com | stream.       | 443         | 443            | false              |
# |             |             |                           | binancefuture |             |                |                    |
# |             |             |                           | .com          |             |                |                    |
# +-------------+-------------+---------------------------+---------------+-------------+----------------+--------------------+
# | COINM       | Live        | dapi.binance.com          | dstream.      | 443         | 443            | true               |
# |             |             |                           | binance.com   |             |                |                    |
# |             +-------------+---------------------------+---------------+-------------+----------------+--------------------+
# |             | Testnet     | testnet.binancefuture.com | dstream.      | 443         | 443            | false              |
# |             |             |                           | binancefuture |             |                |                    |
# |             |             |                           | .com          |             |                |                    |
# +-------------+-------------+---------------------------+---------------+-------------+----------------+--------------------+
# | SPOT        | Live        | api.binance.com           | stream.       | 443         | 9443           | true               |
# |             |             |                           | binance.com   |             |                |                    |
# |             +-------------+---------------------------+---------------+-------------+----------------+--------------------+
# |             | Testnet     | testnet.binance.vision    | testnet.      | 443         | 443            | false              |
# |             |             |                           | binance.      |             |                |                    |
# |             |             |                           | vision        |             |                |                    |
# +-------------+-------------+---------------------------+---------------+-------------+----------------+--------------------+
#
# Additional Configuration Notes:
# 1. Authentication:
#    - API Key and Secret Key required for authenticated endpoints
#    - Keys can be provided via:
#      a) Direct string parameters
#      b) Key file (3-line format: environment, API key, secret key)
#
# 2. Key File Format:
#    - Line 1: "live" or "test" (environment indicator)
#    - Line 2: API key
#    - Line 3: Secret key
#    - Maximum file size: 140 bytes
#
# 3. Default Values:
#    - Default REST Port: "443"
#    - Default WS Port: "443" (except SPOT Live: "9443")
#    - Default check_cert: true
#    - Default verifyPeer: true (except testnet: false)
#
# 4. API Documentation References:
#    - Unified docs: https://binance-docs.github.io/apidocs/#general-info
#    - Spot API: https://developers.binance.com/docs/binance-spot-api-docs/README
#    - Margin API: https://developers.binance.com/docs/margin_trading/Introduction
#    - Derivatives API: https://developers.binance.com/docs/derivatives/Introduction
#    - Testnet API: https://binance-docs.github.io/apidocs/testnet/en/#general-info
#    - Public Data API: https://github.com/binance/binance-public-data

# Create all necessary directories
mkdir -p "${SPOT_DIR}/zip" "${SPOT_DIR}/csv"
mkdir -p "${UM_DIR}/zip" "${UM_DIR}/csv"
mkdir -p "${CM_DIR}/zip" "${CM_DIR}/csv"
mkdir -p "$SYMBOL_TABLE_DIR"

# Download symbol lists
curl -x $http_proxy -s -H 'Content-Type: application/json' https://api.binance.com/api/v3/exchangeInfo | \
    jq -r '.symbols | sort_by(.symbol) | .[] | .symbol' > "${SYMBOL_TABLE_DIR}/spot_symbols.txt"

curl -x $http_proxy -s -H 'Content-Type: application/json' https://fapi.binance.com/fapi/v1/exchangeInfo | \
    jq -r '.symbols | sort_by(.symbol) | .[] | .symbol' > "${SYMBOL_TABLE_DIR}/um_symbols.txt"

curl -x $http_proxy -s -H 'Content-Type: application/json' https://dapi.binance.com/dapi/v1/exchangeInfo | \
    jq -r '.symbols | sort_by(.symbol) | .[] | .symbol' > "${SYMBOL_TABLE_DIR}/cm_symbols.txt"

# Function to check if a CSV file is already processed and valid
is_csv_processed() {
    local csv_path=$1
    if [ -f "$csv_path" ] && [ -s "$csv_path" ] && [ $(wc -l < "$csv_path") -gt 0 ]; then
        return 0
    fi
    return 1
}

# Function to extract ZIP file to CSV
extract_to_csv() {
    local zip_path=$1
    local csv_dir=$2
    local filename=$(basename "$zip_path" .zip)
    local csv_path="${csv_dir}/${filename}.csv"
    
    if is_csv_processed "$csv_path"; then
        echo "CSV already exists: ${csv_path}"
        return 0
    fi
    
    mkdir -p "$csv_dir"
    
    if unzip -q "$zip_path" -d "$csv_dir"; then
        if [ -f "${csv_dir}/${filename}.csv" ]; then
            return 0
        fi
        echo "Error extracting: ${zip_path}"
        return 1
    else
        echo "Error extracting: ${zip_path}"
        return 1
    fi
}

# Function to download a single file
download_file() {
    local base_url=$1
    local market_type=$2  # "spot", "um", or "cm"
    local symbol=$3
    local interval=$4
    local year=$5
    local month=$6
    
    local filename="${symbol}-${interval}-${year}-${month}.zip"
    local zip_dir="${BASE_DIR}/${market_type}/zip/${symbol}/${interval}"
    local csv_dir="${BASE_DIR}/${market_type}/csv/${symbol}/${interval}"
    local zip_path="${zip_dir}/${filename}"
    local full_url="${base_url}/${symbol}/${interval}/${filename}"
    
    # Print exact URL being accessed
    # echo "Accessing URL: ${full_url}"
    # echo "Target ZIP: ${zip_path}"
    
    mkdir -p "$zip_dir" "$csv_dir"
    
    local csv_path="${csv_dir}/$(basename "$filename" .zip).csv"
    if [ -f "$zip_path" ] && [ -s "$zip_path" ] && is_csv_processed "$csv_path"; then
        echo "Skipping existing file: ${filename}"
        return 0
    fi
    
    response=$(curl -x $http_proxy -o "$zip_path" -w "%{http_code}" -s "$full_url")
    
    case $response in
        200)
            echo "Downloaded (${response}): ${full_url}"
            # echo "Saved to: ${zip_path}"
            extract_to_csv "$zip_path" "$csv_dir"
            return 0
            ;;
        404)
            echo "File not found (${response}): ${full_url}"
            rm -f "$zip_path"
            return 1
            ;;
        *)
            echo "Error (HTTP ${response}): ${full_url}"
            rm -f "$zip_path"
            return 1
            ;;
    esac
}

# Function to process downloads for spot market
process_spot_downloads() {
    if [ "$SPOT_ENABLED" = true ]; then
        echo "Starting spot market downloads..."
        echo "Base URL: ${SPOT_BASE_URL}"
        
        for symbol in "${SPOT_SYMBOLS[@]}"; do
            echo "Processing symbol: ${symbol}"
            for interval in "${INTERVALS[@]}"; do
                echo "Processing interval: ${interval}"
                echo "Full path pattern: ${SPOT_BASE_URL}/${symbol}/${interval}/${symbol}-${interval}-YYYY-MM.zip"
                
                for year in "${YEARS[@]}"; do
                    for month in "${MONTHS[@]}"; do
                        download_file "${SPOT_BASE_URL}" "spot" "$symbol" "$interval" "$year" "$month" &
                        if [ $(jobs -r | wc -l) -ge 5 ]; then
                            wait -n
                        fi
                    done
                    wait
                done
            done
        done
    fi
}

# Function to process downloads for futures market
process_futures_downloads() {
    if [ "$FUTURES_ENABLED" = true ]; then
        echo "Starting ${CM_OR_UM} futures market downloads..."
        echo "Base URL: ${FUTURES_BASE_URL}"
        
        for symbol in "${FUTURES_SYMBOLS[@]}"; do
            echo "Processing symbol: ${symbol}"
            for interval in "${INTERVALS[@]}"; do
                echo "Processing interval: ${interval}"
                echo "Full path pattern: ${FUTURES_BASE_URL}/${symbol}/${interval}/${symbol}-${interval}-YYYY-MM.zip"
                
                for year in "${YEARS[@]}"; do
                    for month in "${MONTHS[@]}"; do
                        download_file "${FUTURES_BASE_URL}" "${CM_OR_UM}" "$symbol" "$interval" "$year" "$month" &
                        if [ $(jobs -r | wc -l) -ge 5 ]; then
                            wait -n
                        fi
                    done
                    wait
                done
            done
        done
    fi
}

# Execute downloads
process_spot_downloads
process_futures_downloads

echo "All downloads and extractions completed!"

# Final directory structure will be:
# downloads/
# ├── spot/
# │   ├── zip/
# │   │   └── [symbol]/[interval]/[files.zip]
# │   └── csv/
# │       └── [symbol]/[interval]/[files.csv]
# ├── um/
# │   ├── zip/
# │   │   └── [symbol]/[interval]/[files.zip]
# │   └── csv/
# │       └── [symbol]/[interval]/[files.csv]
# ├── cm/
# │   ├── zip/
# │   │   └── [symbol]/[interval]/[files.zip]
# │   └── csv/
# │       └── [symbol]/[interval]/[files.csv]
# └── symbols/
#     ├── spot_symbols.txt
#     ├── um_symbols.txt
#     └── cm_symbols.txt