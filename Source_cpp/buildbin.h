#ifndef BUILDBIN_H
#define BUILDBIN_H

#include <armadillo>
#include <boost/python.hpp>
#include <list>

class Bin {
  private:
   arma::vec x=arma::zeros(2000);
   arma::mat grid=arma::zeros(2000,2);  // The class
  public:           // Access specifier
    Bin(arma::vec x1, arma::mat grid1) {     // Constructor
      x=x1;
      grid=grid1;
    }
    arma::vec get_bin() {return x;};
    arma::mat get_grid() {return grid;};
};


Bin zerobin1(arma::vec bi,arma::vec gr1);

Bin buildbinC(arma::vec x, int R);

boost::python::dict buildbin(boost::python::list x,int R);

std::list<Bin> buildbinmargC(arma::mat X, arma::vec R);

boost::python::list buildbinmarg(boost::python::list x,boost::python::list R);

#endif /* BUILDBIN_H */