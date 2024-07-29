// g++ -std=c++17 -O3 -pthread main.cpp -o stock_processor -I/path/to/rapidjson/include

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

namespace fs = std::filesystem;

struct StockData {
    std::chrono::system_clock::time_point datetime;
    std::string code;
    double open, high, low, close;
    int64_t volume;
    double amount;
};

struct IntegrityCheck {
    bool integrityFlag;
    std::vector<uint8_t> rulesViolated;
};

struct Metadata {
    std::string assetName;
    std::vector<bool> yearlyDataIntegrity;
    std::string startDate, endDate, firstTraded, autoCloseDate;
    std::string exchange, subExchange;
    std::string industrySectorLevel1, industrySectorLevel2;
    std::string reserved;
};

std::unordered_map<std::string, std::vector<StockData>> globalStockData;
std::mutex globalMutex;
std::atomic<size_t> processedFiles(0);
size_t totalFiles = 0;

void processCSVFile(const fs::path& filePath) {
    std::ifstream file(filePath);
    std::string line;
    std::getline(file, line); // Skip header

    std::vector<StockData> localStockData;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        StockData data;

        std::getline(iss, token, ',');
        std::string dateStr = token;
        std::getline(iss, token, ',');
        std::string timeStr = token;
        std::string datetimeStr = dateStr + " " + timeStr;
        
        std::tm tm = {};
        std::istringstream ss(datetimeStr);
        ss >> std::get_time(&tm, "%Y/%m/%d %H:%M");
        data.datetime = std::chrono::system_clock::from_time_t(std::mktime(&tm));

        std::getline(iss, data.code, ',');
        std::getline(iss, token, ','); data.open = std::stod(token);
        std::getline(iss, token, ','); data.high = std::stod(token);
        std::getline(iss, token, ','); data.low = std::stod(token);
        std::getline(iss, token, ','); data.close = std::stod(token);
        std::getline(iss, token, ','); data.volume = std::stoll(token);
        std::getline(iss, token, ','); data.amount = std::stod(token);

        localStockData.push_back(data);
    }

    std::lock_guard<std::mutex> lock(globalMutex);
    for (const auto& data : localStockData) {
        globalStockData[data.code].push_back(data);
    }

    processedFiles++;
    if (processedFiles % 100 == 0) {
        std::cout << "\rProcessed " << processedFiles << " / " << totalFiles << " files" << std::flush;
    }
}

void importCSVData(const std::string& dataDir) {
    std::vector<std::thread> threads;
    for (const auto& entry : fs::recursive_directory_iterator(dataDir)) {
        if (entry.path().extension() == ".csv") {
            totalFiles++;
        }
    }

    for (const auto& entry : fs::recursive_directory_iterator(dataDir)) {
        if (entry.path().extension() == ".csv") {
            threads.emplace_back(processCSVFile, entry.path());
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "\nFinished processing all files.\n";
}

IntegrityCheck checkIntegrity(const std::vector<StockData>& data) {
    IntegrityCheck check;
    check.integrityFlag = true;

    for (size_t i = 0; i < data.size(); ++i) {
        // Rule 0: Non-zero OHLC
        if (data[i].open == 0 || data[i].high == 0 || data[i].low == 0 || data[i].close == 0) {
            check.rulesViolated.push_back(0);
            check.integrityFlag = false;
        }

        // Rule 1: Timestamp continuity/order/completeness
        if (i > 0) {
            auto diff = std::chrono::duration_cast<std::chrono::minutes>(data[i].datetime - data[i-1].datetime).count();
            if (diff != 1) {
                check.rulesViolated.push_back(1);
                check.integrityFlag = false;
            }
        }

        // Rule 2: Intra-day price continuity
        if (data[i].high < data[i].low || data[i].open > data[i].high || data[i].open < data[i].low || 
            data[i].close > data[i].high || data[i].close < data[i].low) {
            check.rulesViolated.push_back(2);
            check.integrityFlag = false;
        }

        // Rule 3: Inter-day price jump limit
        if (i > 0) {
            double prevClose = data[i-1].close;
            if (std::abs(data[i].open - prevClose) / prevClose > 0.1) {
                check.rulesViolated.push_back(3);
                check.integrityFlag = false;
            }
        }

        // Rule 4: OHLC differ if volume is non-zero
        if (data[i].volume > 0 && data[i].open == data[i].high && data[i].high == data[i].low && data[i].low == data[i].close) {
            check.rulesViolated.push_back(4);
            check.integrityFlag = false;
        }
    }

    return check;
}

std::string getSubExchange(const std::string& code) {
    if (code.substr(0, 2) == "SH") {
        if (code.substr(3, 2) == "60") return "SSE.A";
        if (code.substr(3, 3) == "900") return "SSE.B";
        if (code.substr(3, 2) == "68") return "SSE.STAR";
    } else if (code.substr(0, 2) == "SZ") {
        if (code.substr(3, 3) == "000" || code.substr(3, 3) == "001") return "SZSE.A";
        if (code.substr(3, 3) == "200") return "SZSE.B";
        if (code.substr(3, 3) == "300" || code.substr(3, 3) == "301") return "SZSE.SB";
        if (code.substr(3, 3) == "002" || code.substr(3, 3) == "003") return "SZSE.A";
    }
    if (code.substr(3, 3) == "440" || code.substr(3, 3) == "430" || 
        code.substr(3, 2) == "83" || code.substr(3, 2) == "87") return "NQ";
    return "Unknown";
}

Metadata getMetadata(const std::vector<StockData>& data, const std::string& code) {
    Metadata meta;
    meta.assetName = "Asset_" + code; // Placeholder
    meta.yearlyDataIntegrity = std::vector<bool>(32, true); // Placeholder
    meta.startDate = std::to_string(std::chrono::system_clock::to_time_t(data.front().datetime));
    meta.endDate = std::to_string(std::chrono::system_clock::to_time_t(data.back().datetime));
    meta.firstTraded = meta.startDate;
    meta.autoCloseDate = std::to_string(std::chrono::system_clock::to_time_t(data.back().datetime) + 86400); // +1 day
    meta.exchange = code.substr(0, 2);
    meta.subExchange = getSubExchange(code);
    meta.industrySectorLevel1 = "Placeholder";
    meta.industrySectorLevel2 = "Placeholder";
    meta.reserved = "";
    return meta;
}

void saveToJSON(const std::unordered_map<std::string, IntegrityCheck>& integrityTable, 
                const std::unordered_map<std::string, Metadata>& metadataTable) {
    rapidjson::Document integrityDoc;
    integrityDoc.SetObject();
    rapidjson::Document::AllocatorType& allocator = integrityDoc.GetAllocator();

    for (const auto& pair : integrityTable) {
        rapidjson::Value key(pair.first.c_str(), allocator);
        rapidjson::Value obj(rapidjson::kObjectType);
        obj.AddMember("integrityFlag", pair.second.integrityFlag, allocator);
        
        rapidjson::Value rulesArray(rapidjson::kArrayType);
        for (const auto& rule : pair.second.rulesViolated) {
            rulesArray.PushBack(rule, allocator);
        }
        obj.AddMember("rulesViolated", rulesArray, allocator);

        integrityDoc.AddMember(key, obj, allocator);
    }

    rapidjson::StringBuffer integrityBuffer;
    rapidjson::Writer<rapidjson::StringBuffer> integrityWriter(integrityBuffer);
    integrityDoc.Accept(integrityWriter);

    std::ofstream integrityFile("integrity_table.json");
    integrityFile << integrityBuffer.GetString();
    integrityFile.close();

    rapidjson::Document metadataDoc;
    metadataDoc.SetObject();

    for (const auto& pair : metadataTable) {
        rapidjson::Value key(pair.first.c_str(), allocator);
        rapidjson::Value obj(rapidjson::kObjectType);
        obj.AddMember("assetName", rapidjson::Value(pair.second.assetName.c_str(), allocator), allocator);
        
        rapidjson::Value yearlyIntegrityArray(rapidjson::kArrayType);
        for (const auto& integrity : pair.second.yearlyDataIntegrity) {
            yearlyIntegrityArray.PushBack(integrity, allocator);
        }
        obj.AddMember("yearlyDataIntegrity", yearlyIntegrityArray, allocator);

        obj.AddMember("startDate", rapidjson::Value(pair.second.startDate.c_str(), allocator), allocator);
        obj.AddMember("endDate", rapidjson::Value(pair.second.endDate.c_str(), allocator), allocator);
        obj.AddMember("firstTraded", rapidjson::Value(pair.second.firstTraded.c_str(), allocator), allocator);
        obj.AddMember("autoCloseDate", rapidjson::Value(pair.second.autoCloseDate.c_str(), allocator), allocator);
        obj.AddMember("exchange", rapidjson::Value(pair.second.exchange.c_str(), allocator), allocator);
        obj.AddMember("subExchange", rapidjson::Value(pair.second.subExchange.c_str(), allocator), allocator);
        obj.AddMember("industrySectorLevel1", rapidjson::Value(pair.second.industrySectorLevel1.c_str(), allocator), allocator);
        obj.AddMember("industrySectorLevel2", rapidjson::Value(pair.second.industrySectorLevel2.c_str(), allocator), allocator);
        obj.AddMember("reserved", rapidjson::Value(pair.second.reserved.c_str(), allocator), allocator);

        metadataDoc.AddMember(key, obj, allocator);
    }

    rapidjson::StringBuffer metadataBuffer;
    rapidjson::Writer<rapidjson::StringBuffer> metadataWriter(metadataBuffer);
    metadataDoc.Accept(metadataWriter);

    std::ofstream metadataFile("metadata_table.json");
    metadataFile << metadataBuffer.GetString();
    metadataFile.close();
}

int main() {
    const std::string dataDir = "path/to/data";

    auto start = std::chrono::high_resolution_clock::now();

    importCSVData(dataDir);

    std::unordered_map<std::string, IntegrityCheck> integrityTable;
    std::unordered_map<std::string, Metadata> metadataTable;

    for (const auto& pair : globalStockData) {
        const std::string& code = pair.first;
        const auto& data = pair.second;

        integrityTable[code] = checkIntegrity(data);
        metadataTable[code] = getMetadata(data, code);
    }

    saveToJSON(integrityTable, metadataTable);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    return 0;
}