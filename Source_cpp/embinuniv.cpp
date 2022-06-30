#include "embinuniv.h"
#include <boost/python/extract.hpp>
#include <boost/python/dict.hpp>
#include <vector>
#include <list>
#include "conversion.h"
#include "buildbin.h"
#include "loglik.h"

namespace p = boost::python;

EmBinUniv emunivC(Bin bin,arma::vec pi0,arma::vec mu0,arma::vec sigma0,double eps,int it)
{
  arma::vec bi=bin.get_bin();
  arma::mat gr=bin.get_grid();
  arma::vec logli(it+1);
  int lenbin=bi.n_elem;
  int lenp=pi0.n_elem;
  int j;
  int n=arma::accu(bi);
  arma::mat s(lenbin,lenp);
  arma::vec pi1(lenp);
  arma::vec mu1(lenp);
  arma::vec sigma1(lenp);
  logli(0)=loglimultC(bin,pi0,mu0,sigma0);
  double z1;
  double z2;
  double z;
  int i;
  int flag=1;
  int k=1;
  while(flag==1)
  {
    arma::mat sm(lenbin,lenp);
    arma::mat sv(lenbin,lenp);
    for(j=0;j<lenp;j++)
    {
      boost::math::normal_distribution<>nd(mu0(j),sqrt(sigma0(j)));
      for(i=0;i<lenbin;i++)
      {
        if(gr(i,0)==-INFINITY)
        {
          sv(i,j)=pdf(nd,gr(i,1))*gr(i,1);
        } else {
          if(gr(i,1)==INFINITY){
            sv(i,j)=-pdf(nd,gr(i,0))*gr(i,0);
          } else {
            sv(i,j)=pdf(nd,gr(i,1))*gr(i,1)-pdf(nd,gr(i,0))*gr(i,0);
          }
        }
        z1=cdf(nd,gr(i,1))-cdf(nd,gr(i,0));
        z2=cdf(complement(nd,gr(i,0)))-cdf(complement(nd,gr(i,1)));
        sm(i,j)=pdf(nd,gr(i,1))-pdf(nd,gr(i,0));
        if(z1>=z2)
        {
          s(i,j)=z1;
        } else
        {
          s(i,j)=z2;
        }
      }
    }
    arma::vec s1(lenbin);
    s1=s*pi0;
    arma::mat inti(lenbin,lenp);
    for(i=0;i<lenbin;i++)
      for(j=0;j<lenp;j++)
        inti(i,j)=s(i,j)*pi0(j)/s1(i);
    arma::mat intm(lenbin,lenp);
    for(i=0;i<lenbin;i++)
      for(j=0;j<lenp;j++)
        intm(i,j)=(pi0(j)/s1(i))*((s(i,j)*mu0(j))-(sm(i,j)*sigma0(j)));
    arma::vec sum1=inti.t()*bi;
    pi1=sum1/n;
    arma::vec sum2=intm.t()*bi;
    mu1=sum2/sum1;
    arma::mat intv(lenbin,lenp);
    for(i=0;i<lenbin;i++)
      for(j=0;j<lenp;j++)
        intv(i,j)=(pi0(j)/s1(i))*((s(i,j)+sm(i,j)*(2*mu1(j)-mu0(j))-sv(i,j))*sigma0(j)+s(i,j)*(mu1(j)-mu0(j))*(mu1(j)-mu0(j)));
    arma::vec sum3=intv.t()*bi;
    sigma1=sum3/sum1;
    logli(k)=loglimultC(bin,pi1,mu1,sigma1);
    if(logli(k) == -std::numeric_limits<double>::infinity())
    {EmBinUniv em(bin,pi1,mu1,sigma1,logli);
    return em;}
    flag=0;
    if ((abs((logli(k) - logli(k - 1))/logli(k - 1)) > eps) &
        (k < it)) {
      flag = 1;
      pi0 = pi1;
      mu0 = mu1;
      sigma0 = sigma1;
      k=k+1;
    }
  }
  arma::vec logli1(k+1);
  for(i=0;i<=k;i++)
    logli1(i)=logli(i);
  EmBinUniv em(bin,pi1,mu1,sigma1,logli1);
  return em;
}