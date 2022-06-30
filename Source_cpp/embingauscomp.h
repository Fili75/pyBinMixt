#ifndef EMBINGAUSCOMP_H
#define EMBINGAUSCOMP_H

#include <armadillo>
#include <boost/python.hpp>
#include <boost/python/dict.hpp>
#include <list>
#include <boost/math/distributions/normal.hpp>
#include "buildbin.h"

class EmBin {
  private:
   std::list<Bin> bin;
   arma::vec pi=arma::zeros(2);
   arma::mat mu=arma::zeros(2,2);
   arma::mat v=arma::zeros(2,2);
   arma::vec logli=arma::zeros(2); // The class
  public:           // Access specifier
    EmBin(std::list<Bin> bin1,arma::vec pi1,arma::mat mu1, arma::mat v1,arma::vec logli1) {     // Constructor
      bin=bin1;
      mu=mu1;
      pi=pi1;
      v=v1;
      logli=logli1;
    }
    std::list<Bin> get_bin()
    {
     return bin;
    }
    arma::vec get_pi()
    {
     return pi;
    }
        arma::mat get_mu()
    {
     return mu;
    }
    arma::mat get_v()
{
 return v;
}
arma::vec get_logli()
{
return logli;
}
};

class EmDim {
  private:
   arma::vec pi=arma::zeros(2);
   arma::vec mu=arma::zeros(2);
   arma::vec v=arma::zeros(2);
  public:           // Access specifier
    EmDim(arma::vec pi1,arma::vec mu1, arma::vec v1) {     // Constructor
      mu=mu1;
      pi=pi1;
      v=v1;
    }
    arma::vec get_pi()
    {
     return pi;
    }
        arma::vec get_mu()
    {
     return mu;
    }
    arma::vec get_v()
{
 return v;
}
};

EmDim emdimC(Bin bin,arma::vec pi0,arma::vec mu0,arma::vec sigma0);

EmBin embingauscompC(std::list<Bin> bin,arma::vec pi0,arma::mat mu0,arma::mat sigma0,double eps, int it);

boost::python::dict embingauscomp(boost::python::list bin,boost::python::list grid,boost::python::list pi0,boost::python::list mu0,boost::python::list sigma0,double eps, int it);

#endif /* EMBINGAUSCOMP_H */