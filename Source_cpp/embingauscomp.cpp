#include "embingauscomp.h"
#include <boost/python/extract.hpp>
#include <boost/python/dict.hpp>
#include <vector>
#include <list>
#include "conversion.h"
#include "buildbin.h"
#include "loglik.h"

namespace p=boost::python;

EmDim emdimC(Bin bin,arma::vec pi0,arma::vec mu0,arma::vec sigma0)
{
  arma::vec bi=bin.get_bin();
  arma::mat gr=bin.get_grid();
  int lenbin=bi.n_elem;
  int lenp=pi0.n_elem;
  int j;
  arma::mat s(lenbin,lenp);
  double z1;
  double z2;
  double z;
  int i;
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
  arma::vec sum2=intm.t()*bi;
  arma::vec mu1=sum2/sum1;
  arma::mat intv(lenbin,lenp);
  for(i=0;i<lenbin;i++)
    for(j=0;j<lenp;j++)
      intv(i,j)=(pi0(j)/s1(i))*((s(i,j)+sm(i,j)*(2*mu1(j)-mu0(j))-sv(i,j))*sigma0(j)+s(i,j)*(mu1(j)-mu0(j))*(mu1(j)-mu0(j)));
  arma::vec sum3=intv.t()*bi;
  arma::vec sigma1=sum3/sum1;
  EmDim e(sum1,mu1,sigma1);
  return e;
}

EmBin embingauscompC(std::list<Bin> bin,arma::vec pi0,arma::mat mu0,arma::mat sigma0,double eps, int it)
{
  int flag=1;
  Bin bin1=bin.front();
  arma::vec bin1bin=bin1.get_bin();
  int n=arma::accu(bin1bin);
  int j=0;
  int linp=pi0.n_elem;
  int dim=mu0.n_rows;
  arma::mat mu1(dim,linp);
  arma::mat sigma1(dim,linp);
  arma::vec pi1(linp);
  int k;
  arma::vec logli(it+1);
  logli(0)=loglimargC(bin,pi0,mu0,sigma0);
  int i;
  while(flag==1) {
    std::list<Bin> bin1=bin;
    pi1=arma::zeros(linp);
    for(i=0;i<dim;i++)
    {
      arma::mat mut=mu0.t();
      arma::mat vt=sigma0.t();
      EmDim dim1=emdimC(bin1.front(),pi0,mut.col(i),vt.col(i));
      bin1.pop_front();
      mu1.row(i)=(dim1.get_mu()).t();
      sigma1.row(i)=(dim1.get_v()).t();
      pi1=pi1+dim1.get_pi();
    }
    pi1=pi1/(dim*n);
    j = j + 1;
    logli(j)=loglimargC(bin,pi1,mu1,sigma1);
    if(logli(j) == -std::numeric_limits<double>::infinity())
    {
      arma::vec logli2(j+1);
      for(i=0;i<=j;i++)
        logli2(i)=logli(i);
      EmBin em(bin,pi1,mu1,sigma1,logli2);
    return em;}
    flag=0;
    if ((abs((logli(j) - logli(j - 1))/logli(j - 1)) > eps) &
        (j < it)) {
      flag = 1;
      pi0 = pi1;
      mu0 = mu1;
      sigma0 = sigma1;
    }
  }
  arma::vec logli2(j+1);
  for(i=0;i<=j;i++)
    logli2(i)=logli(i);
  EmBin em(bin,pi1,mu1,sigma1,logli2);
  return em;
}

p::dict embingauscomp(p::list bin,p::list grid,p::list pi0,p::list mu0,p::list sigma0,double eps, int it)
{
  int l=p::len(bin);
  int i=0;
  std::list<Bin> a;
  for(i=0;i<l;i++)
  {
    arma::vec bv=listTOvec(boost::python::extract<p::list>(bin[i]));
    arma::mat bg=listTOmat(boost::python::extract<p::list>(grid[i]));
    Bin b(bv,bg);
    a.push_back(b);
  }
  arma::vec pi1=listTOvec(pi0);
  arma::mat mu1=listTOmat(mu0);
  arma::mat v1=listTOmat(sigma0);
  EmBin bi=embingauscompC(a,pi1,mu1,v1,eps,it);
  p::dict dic;
  dic["bin"]=bin;
  dic["pi"]=vecTOlist(bi.get_pi());
  dic["mu"]=matTOlist(bi.get_mu());
  dic["v"]=matTOlist(bi.get_v());
  dic["loglik"]=vecTOlist(bi.get_logli());
  return dic;
}