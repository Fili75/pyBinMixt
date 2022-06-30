#ifndef DMIXT_H
#define DMIXT_H

#include <armadillo>
#include <boost/python.hpp>
#include <list>

arma::vec dmixtC(arma::vec x, arma::vec pi, arma::vec mu, arma::vec v);

boost::python::list dmixt(boost::python::list x,boost::python::list pi,boost::python::list mu,boost::python::list v);

#endif /* DMIXT_H */