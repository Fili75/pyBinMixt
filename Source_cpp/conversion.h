#ifndef CONVERSION_H
#define CONVERSION_H

#include <boost/python.hpp>
#include <armadillo>

arma::vec listTOvec(boost::python::list x);
arma::mat listTOmat(boost::python::list x);
boost::python::list vecTOlist(arma::vec x);
boost::python::list matTOlist(arma::mat x);

#endif /* CONVERSION_H */