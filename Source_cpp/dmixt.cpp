#include <armadillo>
#include <boost/python.hpp>
#include <list>
#include "dmixt.h"
#include "conversion.h"
#include <boost/math/distributions/normal.hpp>

namespace p=boost::python;


arma::vec dmixtC(arma::vec x, arma::vec pi, arma::vec mu, arma::vec v)
{
  int len=x.n_elem;
  int lenp=pi.n_elem;
  int i;
  int j;
  arma::vec y=arma::zeros(len);
  double z;
  for(j=0;j<lenp;j++)
  {
    boost::math::normal_distribution<>nd(mu(j),sqrt(v(j)));
    for(i=0;i<len;i++)
      y(i)=pdf(nd,x(i))*pi(j)+y(i);
  }
  return y;
}

p::list dmixt(p::list x,p::list pi,p::list mu,p::list v)
{
  arma::vec x1=listTOvec(x);
  arma::vec pi1=listTOvec(pi);
  arma::mat mu1=listTOvec(mu);
  arma::mat v1=listTOvec(v);
  arma::vec res=dmixtC(x1,pi1,mu1,v1);
  return vecTOlist(res);
}