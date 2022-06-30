#ifndef BINMIXT_H
#define BINMIXT_H

#include <armadillo>
#include <boost/python.hpp>
#include <list>
#include <boost/python/dict.hpp>
#include <boost/math/distributions/normal.hpp>
#include "buildbin.h"
#include "embinuniv.h"
#include "embingauscomp.h"

EmBinUniv binmixtuC(arma::vec data, int cl, arma::vec R, int it, double eps, int nrep);


EmBin binmixtC(arma::mat data, int cl, arma::vec R, int it, double eps, int nrep, int seed=1);


boost::python::dict binmixt(boost::python::list data, int cl, boost::python::list R, int it, double eps, int nrep,int seed=1);

#endif /* BINMIXT_H */