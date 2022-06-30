#ifndef GENMIXT_H
#define GENMIXT_H

#include<armadillo>
#include <boost/python.hpp>
#include <boost/python/dict.hpp>

class Mixt {
  private:
   arma::mat x;
   arma::vec ind;  // The class

  public:           // Access specifier
    Mixt(arma::mat x=arma::zeros(2,2), arma::vec ind=arma::zeros(2)) : x(x), ind(ind) {
    };
    arma::mat get_x()
    {
     return x;
    }
        arma::vec get_ind()
    {
     return ind;
    }
};

int whichmaxC(arma::vec x);

arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma);

Mixt genmixtC(int n, arma::vec p,arma::mat mu, arma::mat sigma, int seed);

boost::python::dict genmixt(int n,boost::python::list p, boost::python::list mu, boost::python::list sigma, int seed);

#endif /* GENMIXT_H */