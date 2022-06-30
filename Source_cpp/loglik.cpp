#include "dmixt.h"
#include "loglik.h"
#include "buildbin.h"
#include "conversion.h"
#include <boost/math/distributions/normal.hpp>

namespace p=boost::python;

double loglimixtC(arma::vec x, arma::vec pi, arma::vec mu, arma::vec v)
{
  int len=x.n_elem;
  int i;
  arma::vec y=dmixtC(x,pi,mu,v);
  double z=0;
  for(i=0;i<len;i++)
  {
    z=z+log(y(i));
  }
  return z;
}

double loglimixt(p::list x,p::list pi,p::list mu,p::list v)
{
  arma::vec x1=listTOvec(x);
  arma::vec pi1=listTOvec(pi);
  arma::vec mu1=listTOvec(mu);
  arma::vec v1=listTOvec(v);
  return loglimixtC(x1,pi1,mu1,v1);
}

double loglimultC(Bin bin, arma::vec pi, arma::vec mu, arma::vec v)
{
  arma::vec bi=bin.get_bin();
  arma::mat gr=bin.get_grid();
  int lenbin=bi.n_elem;
  int lenp=pi.n_elem;
  int j;
  int i;
  arma::vec y(lenbin);
  double z=0;
  double w=0;
  double x;
  for(j=0;j<lenp;j++)
  {
    boost::math::normal_distribution<>nd(mu(j),sqrt(v(j)));
    for(i=0;i<lenbin;i++)
    {
      z=(cdf(nd,gr(i,1))-cdf(nd,gr(i,0)));
      w=cdf(complement(nd,gr(i,0)))-cdf(complement(nd,gr(i,1)));
      if(z>=w)
        y(i)=z*pi(j)+y(i);
      else
        y(i)=w*pi(j)+y(i);
    }
  }
  x=0;
  for(i=0;i<lenbin;i++)
  {
    x=x+bi(i)*log(y(i));
  }
  return x;
}

double loglimult(p::list bin, p::list grid, p::list pi,p::list mu,p::list v)
{
  arma::vec binv=listTOvec(bin);
  arma::mat gridv=listTOmat(grid);
  Bin bin1(binv,gridv);
  arma::vec pi1=listTOvec(pi);
  arma::vec mu1=listTOvec(mu);
  arma::vec v1=listTOvec(v);
  double l=loglimultC(bin1,pi1,mu1,v1);
  return l;
}


double loglimargC(std::list<Bin> bin, arma::vec pi, arma::mat mu, arma::mat v)
{
  int dim=mu.n_cols;
  double z=0;
  int i;
  arma::mat mut=mu.t();
  arma::mat vt=v.t();
  for(i=0;i<dim;i++)
  {
    z=z+loglimultC(bin.front(),pi,mut.col(i),vt.col(i));
    bin.pop_front();
  }
  return z;
}

double loglimarg(p::list bin,p::list grid, p::list pi,p::list mu,p::list v)
{
  int l=p::len(bin);
  arma::vec bv=listTOvec(boost::python::extract<p::list>(bin[0]));
  arma::mat bg=listTOmat(boost::python::extract<p::list>(grid[0]));
  int i=0;
  std::list<Bin> a;
  Bin b(bv,bg);
  a.push_back(b);
  for(i=1;i<l;i++)
  {
    arma::vec bv=listTOvec(boost::python::extract<p::list>(bin[i]));
    arma::mat bg=listTOmat(boost::python::extract<p::list>(grid[i]));
    Bin b(bv,bg);
    a.push_back(b);
  }
  arma::vec pi1=listTOvec(pi);
  arma::mat mu1=listTOmat(mu);
  arma::mat v1=listTOmat(v);
  return loglimargC(a,pi1,mu1,v1);
}