//
// Created by Khurram Javed on 2024-10-17.
//

#ifndef STDOUT_H
#define STDOUT_H
#include <string>
#include <iostream>
#include <iomanip>
#include <map>


namespace ansii
{
    std::string red = "\033[1;31m";
    std::string green = "\033[1;32m";
    std::string yellow = "\033[1;33m";
    std::string blue = "\033[1;34m";
    std::string magenta = "\033[1;35m";
    std::string cyan = "\033[1;36m";
    std::string reset = "\033[0m";

    // Move cursor to the beginning of the line
    std::string move_to_start = "\r";

    // Clear line from cursor to the end of the line
    std::string clear_line = "\033[K";

    // Set background colors
    std::string bg_red = "\033[41m";
    std::string bg_darkred = "\033[41;30m";
    std::string bg_green = "\033[42m";
    std::string bg_yellow = "\033[43m";
    std::string bg_blue = "\033[44m";
    std::string bg_magenta = "\033[45m";

    // Box drawing characters
    std::string box_horizontal = "─";
    std::string box_vertical = "│";
    std::string box_top_left = "┌";
    std::string box_top_right = "┐";
    std::string box_bottom_left = "└";
    std::string box_bottom_right = "┘";

    std::string bold = "\033[1m";
    std::string underline = "\033[4m";
    std::string italics = "\033[3m";


    // Move cursor in the terminal
    std::string move_cursor(int x, int y)
    {
        return "\033[" + std::to_string(y) + ";" + std::to_string(x) + "H";
    }

    // Clear the screen
    std::string clear_screen()
    {
        return "\033[2J";
    }

    void printPercentage(float percentage, int x, int y, int width)
    {
        std::cout << move_cursor(x, y) << std::setw(20) << "Percentage:" << " [";
        int num_hashes = (int)(percentage * (width - 20));
        for (int i = 0; i < width - 20; i++)
        {
            if (i <= num_hashes)
            {
                std::cout << green << "|" << reset;
            }
            else
            {
                std::cout << " ";
            }
        }
        std::cout << "] " << percentage * 100 << "%";
    }

    void print_table(std::map<std::string, std::string> table, int x, int y, std::string title)
    {
        std::cout << move_cursor(x + 15, y) << bg_magenta << bold << "Table: " << title << reset;
        int i = 0;
        for (auto const& [key, val] : table)
        {
            std::cout << move_cursor(x, y + i + 2) << std::left << std::setw(20) << key << std::setw(10) << ":" <<
                std::setw(20) << val;
            i++;
        }
        std::cout << std::right;
    }

    void print_box(int x, int y, int h, int w)
    {
        for (int j = x; j < x + w; j++)
        {
            std::cout << move_cursor(j, y) << box_horizontal;
            std::cout << move_cursor(j, y + h) << box_horizontal;
        }
        for (int j = y + 1; j < y + h; j++)
        {
            std::cout << move_cursor(x, j) << box_vertical;
            std::cout << move_cursor(x + w, j) << box_vertical;
        }

        std::cout << move_cursor(x, y) << box_top_left;
        std::cout << move_cursor(x + w, y) << box_top_right;
        std::cout << move_cursor(x, y + h) << box_bottom_left;
        std::cout << move_cursor(x + w, y + h) << box_bottom_right;
    }

    void printKeyValue(int x, int y, std::string key, std::string value)
    {
        std::cout << move_cursor(x, y) << std::left << std::setw(20) << key << std::setw(10) << ":" << std::setw(20) <<
            value << std::right;
    }

    void printKeyValue(int x, int y, std::string key, float value)
    {
        std::cout << move_cursor(x, y) << std::left << std::setw(20) << key << std::setw(10) << ":" << std::setw(20) <<
            value << std::right;
    }

    void printKeyValue(int x, int y, std::string key, int value)
    {
        std::cout << move_cursor(x, y) << std::left << std::setw(20) << key << std::setw(10) << ":" << std::setw(20) <<
            value << std::right;
    }
}
#endif //STDOUT_H
