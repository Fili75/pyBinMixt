#include "buildbin.h"
#include <boost/python/extract.hpp>
#include <boost/python/dict.hpp>
#include <vector>
#include <list>
#include "conversion.h"

namespace p = boost::python;

Bin zerobin1(arma::vec bi,arma::vec gr1)
{
  int lenbin=bi.n_elem;
  int dimgr=gr1.n_elem;
  arma::mat gr(dimgr+2,2);
  int i;
  int j;
  int conta=0;
  for(i=0;i<lenbin;i++)
  {
    if(bi(i)>0) conta=conta+1;
  }
  gr(0,0)=-INFINITY;
  gr(dimgr+1,1)=INFINITY;
  for(i=1;i<dimgr+1;i++)
    gr(i,0)=gr1(i-1);
  for(i=0;i<dimgr;i++)
    gr(i,1)=gr1(i);

  arma::mat gr2(conta,2);
  arma::vec bizero(conta);
  conta=0;
  for(i=0;i<lenbin;i++)
  {
    if(bi(i)>0)
    {
      gr2(conta,0)=gr(i,0);
      gr2(conta,1)=gr(i,1);
      bizero(conta)=bi(i);
      conta=conta+1;
    }
  }
  Bin bin1(bizero,gr2);
  return bin1;

}

Bin buildbinC(arma::vec x, int R){
  double maxim=max(x);
  double minim=min(x);
  int len=x.n_elem;
  double interval= (maxim-minim)/(R-1);
  arma::vec z(R);
  z(0)=minim;
  z(R-1)=maxim;
  int i;
  int j;
  for(i=1;i<R-1;i++)
  {
    z(i)=minim+interval*(i);
  }
  arma::vec y=arma::zeros(R+1);
  for(i=0;i<R;i++)
  {
    for(j=0;j<len;j++)
    {
      if(i==0)
      {
        if(x(j)<=z(i)) y(i)=y(i)+1;
      } else {
        if( (x(j)<=z(i)) & (x(j)>z(i-1)))
          y(i)=y(i)+1;}
    }
  }
  return zerobin1(y,z);
}

boost::python::dict buildbin(p::list x,int R)
{
  arma::vec x1=listTOvec(x);
  Bin bin=buildbinC(x1,R);
  boost::python::dict dic;
  dic["bin"]=vecTOlist(bin.get_bin());
  dic["grid"]=matTOlist(bin.get_grid());
  return dic;
}

std::list<Bin> buildbinmargC(arma::mat X, arma::vec R)
{
  int i=0;
  int ncol=X.n_cols;
  std::list<Bin> a;
  for(i=0;i<ncol;i++)
  {
    a.push_back(buildbinC(X.col(i),R(i)));
  }
  return a;
}

p::list buildbinmarg(p::list x,p::list R)
{
  arma::mat X=listTOmat(x);
  arma::vec R1=listTOvec(R);
  std::list<Bin> li=buildbinmargC(X,R1);
  int d=p::len(R);
  int i;
  p::list lip;
  for(i=0;i<d;i++)
  {
    Bin bi=li.front();
    li.pop_front();
    boost::python::dict dic;
    dic["bin"]=vecTOlist(bi.get_bin());
    dic["grid"]=matTOlist(bi.get_grid());
    lip.append(dic);
  }
  return(lip);
}