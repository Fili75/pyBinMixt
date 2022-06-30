#include <boost/math/distributions/normal.hpp>
#include <list>
#include "conversion.h"
#include "posterior.h"

namespace p=boost::python;

arma::mat posteriorC(arma::mat data,arma::vec pi,arma::mat mu, arma::mat v)
{
  int len=data.n_rows;
  int lenp=pi.n_elem;
  int dim=data.n_cols;
  arma::mat predm=arma::ones(len,lenp);
  int i,j,k;
    for(i=0;i<lenp;i++)
    predm.col(i)=pi(j)*arma::ones(len);
  for(j=0;j<lenp; j++)
    {
      for(k=0;k<dim;k++)
      {
      boost::math::normal_distribution<>nd(mu(k,j),sqrt(v(k,j)));
      for(i=0;i<len;i++)
      {
      predm(i,j)=predm(i,j)*pdf(nd,data(i,k));
      }
      }
    }
  return predm;
}

p::list posterior(p::list data,p::list pi,p::list mu, p::list v)
{
arma::mat dataC=listTOmat(data);
arma::vec piC=listTOvec(pi);
arma::mat muC=listTOmat(mu);
arma::mat vC=listTOmat(v);
return matTOlist(posteriorC(dataC,piC,muC,vC));
}