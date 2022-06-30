#ifndef POSTERIOR_H
#define POSTERIOR_H

#include <armadillo>
#include <boost/python.hpp>
#include <list>

arma::mat posteriorC(arma::mat data,arma::vec pi,arma::mat mu, arma::mat v);
boost::python::list posterior(boost::python::list data,boost::python::list pi,boost::python::list mu, boost::python::list v);

#endif /* POSTERIOR_H */