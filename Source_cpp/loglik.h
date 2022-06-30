#ifndef LOGLIK_H
#define LOGLIK_H

#include <armadillo>
#include <boost/python.hpp>
#include <list>
#include "buildbin.h"


double loglimixtC(arma::vec x, arma::vec pi, arma::vec mu, arma::vec v);

double loglimixt(boost::python::list x,boost::python::list pi,boost::python::list mu,boost::python::list v);

double loglimultC(Bin bin, arma::vec pi, arma::vec mu, arma::vec v);

double loglimult(boost::python::list bin, boost::python::list grid, boost::python::list pi,boost::python::list mu,boost::python::list v);

double loglimargC(std::list<Bin> bin, arma::vec pi, arma::mat mu, arma::mat v);

double loglimarg(boost::python::list bin,boost::python::list grid, boost::python::list pi,boost::python::list mu,boost::python::list v);

#endif /* LOGLIK_H */