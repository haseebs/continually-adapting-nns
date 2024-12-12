//
// Created by Khurram Javed on 2024-11-04.
//

#ifndef LOGGER_H
#define LOGGER_H

#include <string>



class Logger
{
public:
    virtual void log(std::string json) = 0;
};

class FileLogger : public Logger
{
    std::string filename;

public:
    FileLogger(std::string filename);
    void log(std::string json) override;
};

class MongoDBLogger : public Logger
{
    std::string uri;
    std::string collection_name;
    std::string db_name;

public:
    MongoDBLogger(std::string uri, std::string collection_name, std::string db_name);
    void log(std::string json) override;
};

#endif //LOGGER_H
