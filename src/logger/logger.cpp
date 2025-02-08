//
// Created by Khurram Javed on 2024-11-04.
//

#include "../../include/logger/logger.h"


#include <bsoncxx/json.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/uri.hpp>
#include <mongocxx/exception/exception.hpp>
#include <fstream>
#include <iostream>

FileLogger::FileLogger(std::string filename)
{
    this->filename = filename;
}

void FileLogger::log(std::string json)
{
    std::ofstream file;
    file.open(this->filename, std::ios_base::app);
    file << json << std::endl;
    file.close();
}

MongoDBLogger::MongoDBLogger(std::string uri, std::string collection_name, std::string db_name)
{

    this->uri = uri;
    this->collection_name = collection_name;
    this->db_name = db_name;

}

void MongoDBLogger::log(std::string json)
{
    std::cout << "Logging to MongoDB" << std::endl;
    std::cout << "URI: " << this->uri << std::endl;
    try {
        auto uri = mongocxx::uri{this->uri};
        std::cout << "URI created" << std::endl;
        mongocxx::client client{uri};
        std::cout << "URI: " << this->uri << std::endl;
        mongocxx::database db = client[this->db_name];
        std::cout << "Database name: " << this->db_name << std::endl;
        mongocxx::collection collection = db[this->collection_name];
        std::cout << "Collection name: " << this->collection_name << std::endl;
        collection.insert_one(bsoncxx::from_json(json));
    } catch (const mongocxx::exception& e) {
        std::cerr << "MongoDB connection error: " << e.what() << std::endl;
        throw; // Re-throw to allow caller to handle
    }
    catch (const std::exception& e) {
        std::cerr << "General error occurred: " << e.what() << std::endl;
        throw; // Re-throw to allow caller to handle
    }
    catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        throw; // Re-throw to allow caller to handle
    }

    // std::cout << "MongoDB commented out for now" << std::endl;
}

