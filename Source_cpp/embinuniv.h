#ifndef EMBINUNIV_H
#define EMBINUNIV_H

#include <armadillo>
#include <boost/python.hpp>
#include <list>
#include <boost/math/distributions/normal.hpp>
#include "buildbin.h"

class EmBinUniv
{
  private:
   arma::vec pi_u=arma::zeros(2);
   arma::vec mu_u=arma::zeros(2);
   arma::vec v_u=arma::zeros(2);
   arma::vec logli_u=arma::zeros(2);
   std::list<Bin> bin_l;
   Bin bin_u=bin_l.front();
  public:           // Access specifier
    EmBinUniv(Bin bin1,arma::vec pi1,arma::vec mu1, arma::vec v1,arma::vec logli1)
    {
      bin_u=bin1;
      mu_u=mu1;
      pi_u=pi1;
      v_u=v1;
      logli_u=logli1;
    }
    Bin get_bin_u()
    {
     return bin_u;
    }
    arma::vec get_pi_u()
    {
     return pi_u;
    }
        arma::vec get_mu_u()
    {
     return mu_u;
    }
    arma::vec get_v_u()
{
 return v_u;
}
arma::vec get_logli_u()
{
return logli_u;
}
};

EmBinUniv emunivC(Bin bin,arma::vec pi0,arma::vec mu0,arma::vec sigma0,double eps,int it);

#endif /* EMBINUNIV_H */