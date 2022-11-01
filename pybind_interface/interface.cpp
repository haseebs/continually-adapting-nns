#include <pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "../include/experiment/Metric.h"
#include "../include/experiment/Experiment.h"

namespace py = pybind11;

PYBIND11_MODULE(FlexibleNN, m) {

    py::class_<Metric>(m, "Metric")
        .def(py::init<std::string, std::string, std::vector<std::string>, std::vector<std::string>, std::vector<std::string>>())
        .def("add_value", &Metric::add_value)
        .def("add_values", &Metric::add_values);

    py::class_<Database>(m, "Database")
        .def(py::init<>())
        .def("create_database", &Database::create_database);

    py::class_<ExperimentJSON>(m, "ExperimentJSON")
        .def(py::init([](std::vector<std::string> args) {
          std::vector<char *> cstrs;
          for (auto &s : args) cstrs.push_back(s.data());
          //for (auto k : cstrs) std::cout << k << std::endl;
          return ExperimentJSON(cstrs.size(), cstrs.data());
          }))
        .def_readonly("database_name", &ExperimentJSON::database_name)
        .def("get_float_param", &ExperimentJSON::get_float_param)
        .def("get_int_param", &ExperimentJSON::get_int_param)
        .def("get_string_param", &ExperimentJSON::get_string_param)
        .def("get_vector_param", &ExperimentJSON::get_vector_param);
}
