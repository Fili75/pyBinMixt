#include "genmixt.h"
#include "conversion.h"
#include <boost/python/dict.hpp>

namespace p=boost::python;

int whichmaxC(arma::vec x)
{
  int i;
  int lenx=x.n_elem;
  int index=0;
  double maxi=x(0);
  for(i=1;i<lenx;i++)
  {
    if(x(i)>maxi)
    {maxi=x(i);
      index=i;
    }
  }
  return index;
}

arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  arma::mat Z(n,ncols);
  Z=Z.t();
  int i;
  for(i=0;i<n;i++)
    Z.col(i)=mu;
  Z=Z.t();
  return Y * chol(sigma)+Z;
}



Mixt genmixtC(int n, arma::vec p,arma::mat mu, arma::mat sigma, int seed) {
  arma::arma_rng::set_seed(seed);
  int lenp=p.n_elem;
  int ncols=mu.n_rows;
  arma::vec ind1(lenp);
  arma::vec ind(n);
  arma::mat Y(n,ncols);
  arma::vec pis=arma::cumsum(arma::sort(p));
  arma::vec prop=arma::randu(n);
  arma::mat var(ncols,ncols);
  int j;
  int i;
  for(i=0; i<n; i++)
  {

    if(prop(i)<pis(0))
    {
     var=arma::diagmat(sigma.col(0));
      ind(i)=1;
    Y.row(i)=mvrnormArma(1, mu.col(0), var);
    }
     else
     {
		   for(j=1;j<lenp;j++)
   			{
				if(prop(i)>=pis(j-1) & prop(i)<pis(j))
				{
					var=arma::diagmat(sigma.col(j));
					ind(i)=j+1;
				    Y.row(i)=mvrnormArma(1, mu.col(j), var);}
			}

     }
   }
  Mixt m(Y,ind);
  return m;
}


boost::python::dict genmixt(int n,p::list p, p::list mu, p::list sigma, int seed)
{
 arma::vec pi=listTOvec(p);
 arma::mat muC=listTOmat(mu);
 arma::mat sigmaC=listTOmat(sigma);
 Mixt m=genmixtC(n,pi,muC,sigmaC,seed);
 arma::mat x=m.get_x();
 arma::vec ind=m.get_ind();
 boost::python::dict dic;
 dic["mixture"]=matTOlist(x);
 dic["ind"]=vecTOlist(ind);
 return dic;
}