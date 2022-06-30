#ifndef WINDOW_H
#define WINDOW_H

#include <armadillo>
#include <boost/python.hpp>
#include <list>


arma::mat windowC(arma::mat data,arma::vec labels, bool sd, bool m, bool rms,int fin);

boost::python::list window(boost::python::list data, boost::python::list labels,int fin, boost::python::object sd, boost::python::object m,boost::python::object rms);

#endif /* WINDOW_H */