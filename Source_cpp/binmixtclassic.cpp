#include "binmixtclassic.h"
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

EmBinUniv binmixtclassicuC(arma::vec data, int cl, arma::vec R, int it, double eps, double eps1, int it1,int nrep)
{
    std::cout<<"Choosing the best initial guess..."<< std::endl;

    EmBinUniv migliore=binmixtuC(data,cl,R,it,eps,nrep);
    std::cout<<"Estimation phase...";
      EmBinUniv e=emunivC(migliore.get_bin_u(),migliore.get_pi_u(),
                     migliore.get_mu_u(),migliore.get_v_u(),eps1,it1);
      std::cout<<"Done."<< std::endl;
      return e;
}
EmBin binmixtclassicC(arma::mat data, int cl, arma::vec R, int it, double eps, double eps1, int it1,int nrep, int seed)
{
    std::cout<<"Choosing the best initial guess..."<< std::endl;

    EmBin migliore=binmixtC(data,cl,R,it,eps,nrep,seed);
    std::cout<<"Estimation phase...";

    EmBin e=embingauscompC(migliore.get_bin(),migliore.get_pi(),
                   migliore.get_mu(),migliore.get_v(),eps1,it1);
    std::cout<<"Done."<< std::endl;
    return e;
}

p::dict binmixtclassic(p::list data, int cl, p::list R, int it, double eps, double eps1, int it1, int nrep, int seed)
{
  int row=p::len(data);
  int col= p::len(boost::python::extract<p::list>(data[0]));
  p::str f="float";
  p::str d="double";
  if(col==1)
  {
    arma::vec dataC=listTOvec(data);
    arma::vec  R1=listTOvec(R);
    EmBinUniv bi=binmixtclassicuC(dataC,cl,R1,it,eps,eps1,it1,nrep);
    p::dict dic;
    boost::python::dict dic1;
    Bin li=bi.get_bin_u();
    dic["bin"]=li;
    dic["pi"]=vecTOlist(bi.get_pi_u());
    dic["mu"]=matTOlist(bi.get_mu_u());
    dic["v"]=matTOlist(bi.get_v_u());
    dic["loglik"]=vecTOlist(bi.get_logli_u());
    return dic;
  }
  else
  {
    int l=col;
    arma::mat dataC=listTOmat(data);
    arma::vec  R1=listTOvec(R);
    EmBin bi=binmixtclassicC(dataC,cl,R1,it,eps,eps1,it1,nrep,seed);
   // arma::mat post=posteriorC(bi,dataC);
    p::dict dic;
    boost::python::dict dic1;
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
    //dic["posterior"]=matTOlist(post);
    return dic;
  }
}