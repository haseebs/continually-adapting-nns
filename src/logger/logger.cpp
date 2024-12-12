//
// Created by Khurram Javed on 2024-11-04.
//

#include "../../include/logger/logger.h"


//#include <bsoncxx/json.hpp>
//#include <mongocxx/client.hpp>
//#include <mongocxx/uri.hpp>
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
//    mongocxx::client client{mongocxx::uri{this->uri}};
//    mongocxx::database db = client[this->db_name];
//    mongocxx::collection collection = db[this->collection_name];
//    collection.insert_one(bsoncxx::from_json(json));
    std::cout << "MongoDB commented out for now" << std::endl;
}

