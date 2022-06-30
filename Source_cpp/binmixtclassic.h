#ifndef BINMIXTCLASSIC_H
#define BINMIXTCLASSIC_H

#include <armadillo>
#include <boost/python.hpp>
#include <list>
#include <boost/python/dict.hpp>
#include <boost/math/distributions/normal.hpp>
#include "embinuniv.h"
#include "embingauscomp.h"

EmBinUniv binmixtclassicuC(arma::vec data, int cl, arma::vec R, int it, double eps, double eps1, int it1,int nrep);
EmBin binmixtclassicC(arma::mat data, int cl, arma::vec R, int it, double eps, double eps1, int it1,int nrep, int seed=1);
boost::python::dict binmixtclassic(boost::python::list data, int cl, boost::python::list R, int it, double eps, double eps1, int it1, int nrep, int seed=1);


#endif /* BINMIXTCLASSIC_H */