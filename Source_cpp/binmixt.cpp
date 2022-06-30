#include "binmixt.h"
#include "embinuniv.h"
#include "embingauscomp.h"
#include <boost/python/extract.hpp>
#include <boost/python/dict.hpp>
#include <vector>
#include <list>
#include "conversion.h"
#include "buildbin.h"
#include "loglik.h"

namespace p=boost::python;

EmBinUniv binmixtuC(arma::vec data, int cl, arma::vec R, int it, double eps, int nrep)
{
    Bin bin=buildbinC(data,R(0));
    int i;
    arma::vec pi0(cl);
    arma::vec mu0(cl);
    arma::vec s0(cl);
    double s;
    int k;
    int j;
    //Progress p(nrep, true);
    arma::vec logli(nrep);
    double minimo;
    double massimo;
    double varianza;
    minimo=arma::min(data.col(0));
    massimo=arma::max(data.col(0));
    varianza=arma::var(data.col(0));
    EmBinUniv e(bin,pi0,mu0,s0,logli);
    double ma=-INFINITY;
    double ma1;
    for(i=0;i<nrep;i++)
    {
      pi0=arma::randu(cl);
      s=0;
      for(k=0;k<cl;k++)
        s=s+pi0(k);
      pi0=pi0/s;
      for(k=0;k<cl;k++)
      {
        double random=arma::randu();
        mu0(k)=(random*(massimo))+(1-random)*minimo;
      }
      for(k=0;k<cl;k++)
      {
        s0(k)=arma::randu()*varianza;
      }
      EmBinUniv e1=emunivC(bin,pi0,mu0,s0,eps,it);
      ma1=arma::max(e1.get_logli_u());
      if(ma1>ma )
      {
        ma=ma1;
        e=e1;
      }
      //p.increment();
    }
    return e;
}


EmBin binmixtC(arma::mat data, int cl, arma::vec R, int it, double eps, int nrep, int seed)
{
  int dim=data.n_cols;
    std::list<Bin> bin=buildbinmargC(data,R);
    int i;
    arma::vec pi0(cl);
    arma::mat mu0(dim,cl);
    arma::mat s0(dim,cl);
    double s;
    int k;
    int j;
    //Progress p(nrep, true);
    arma::vec logli(nrep);
    arma::vec minimo(dim);
    arma::vec massimo(dim);
    arma::vec varianza(dim);
    double ma=-INFINITY;
    double ma1;
    EmBin e(bin,pi0,mu0,s0,logli);
    arma::arma_rng::set_seed(seed);
    for(i=0;i<dim;i++)
    {
      minimo(i)=arma::min(data.col(i));
      massimo(i)=arma::max(data.col(i));
      varianza(i)=arma::var(data.col(i));
    }
    int conto;
    for(i=0;i<nrep;i++)
    {
      pi0=arma::randu(cl);
      s=0;
      for(k=0;k<cl;k++)
        s=s+pi0(k);
      pi0=pi0/s;
      for(k=0;k<cl;k++)
      {
        for(j=0;j<dim;j++)
        {
          double random=arma::randu();
          mu0(j,k)=(random*(massimo(j)))+(1-random)*minimo(j);
        }
      }
      for(k=0;k<cl;k++)
      {
        for(j=0;j<dim;j++)
          s0(j,k)=arma::randu()*varianza(j);
      }
      EmBin e1=embingauscompC(bin,pi0,mu0,s0,eps,it);
      ma1=arma::max(e1.get_logli());
      if(ma1>ma)
      {
        ma=ma1;
        e=e1;
      }
      conto=100*(i+1)/nrep;
      int conto1;
      if(conto%2==0)
      {
        std::cout<<"\033[1;31m|\033[0m";
        for(conto1=0;conto1<conto;conto1++)
        {
        std::cout<<"\033[1;31m=\033[0m";
        }
        for(conto1=conto;conto1<100;conto1++)
        {
        std::cout<<" ";
        }
        printf("\033[1;31m| %d %\033[0m \r",conto);
      }
    }
    std::cout<<"\n";
    return e;
}

p::dict binmixt(p::list data, int cl, p::list R, int it, double eps, int nrep,int seed)
{
  int row=p::len(data);
  int col= p::len(boost::python::extract<p::list>(data[0]));
  if(col==1)
  {
    arma::vec dataC=listTOvec(data);
    arma::vec  R1=listTOvec(R);
    EmBinUniv bi=binmixtuC(dataC,cl,R1,it,eps,nrep);
    Bin li=bi.get_bin_u();
    p::dict dic;
    dic["bin"]=li;
    dic["pi"]=vecTOlist(bi.get_pi_u());
    dic["mu"]=matTOlist(bi.get_mu_u());
    dic["v"]=matTOlist(bi.get_v_u());
    dic["loglik"]=vecTOlist(bi.get_logli_u());

    return dic;
  }
  else
  {
    int l= col;
    arma::mat dataC=listTOmat(data);
    arma::vec  R1=listTOvec(R);
    EmBin bi=binmixtC(dataC,cl,R1,it,eps,nrep,seed);
    p::dict dic;
    std::list<Bin> li=bi.get_bin();
    p::list lip;
    int i;
    for(i=0;i<l;i++)
    {
    p::dict dic1;
    dic1["bin"]=vecTOlist(li.front().get_bin());
    dic1["grid"]=matTOlist(li.front().get_grid());
     lip.append(dic1);
     li.pop_front();
    }
    dic["bin"]=lip;
    dic["pi"]=vecTOlist(bi.get_pi());
    dic["mu"]=matTOlist(bi.get_mu());
    dic["v"]=matTOlist(bi.get_v());
    dic["loglik"]=vecTOlist(bi.get_logli());
    return dic;
  }
}