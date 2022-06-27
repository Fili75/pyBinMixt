#include <iostream>
#include <stdlib.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/extract.hpp>
 #include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/range.hpp>
#include<armadillo>
#include <boost/python/dict.hpp>
#include <bits/stdc++.h>
#include<list>
#include <boost/progress.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;
namespace bm = boost::math;

arma::vec listTOvec(p::list x)
{
 int l=p::len(x);
 int i;
 arma::vec y(l);
 //for(i=0;i<l;i++)
  //  y(i)=boost::python::extract<double>(x[i]);
 std::vector<double> out= std::vector<double>(boost::python::stl_input_iterator<double>(x),
                   boost::python::stl_input_iterator<double>());
 for(i=0;i<l;i++)
 y(i)=out[i];
 return y;
}



arma::mat listTOmat(p::list x)
{
 int row=p::len(x);
 int col=p::len(boost::python::extract<p::list>(x[0]));
 int i;
 int j;
 arma::mat y(row,col);
 std::vector<std::vector<double> > out;
 //for(i=0;i<row;i++)
 //for(j=0;j<col;j++)
  //  y(i,j)=boost::python::extract<double>(x[i][j]);
  for(int i = 0; i < row; i++){
	   out.push_back(std::vector<double>(boost::python::stl_input_iterator<double>(x[i]),
		              boost::python::stl_input_iterator<double>()));
	}
  for(i=0;i<row;i++)
  for(j=0;j<col;j++)
   y(i,j)=out[i][j];
 return y;
}



p::list vecTOlist(arma::vec x)
{
 int l=x.n_elem;
 int i;
 p::list y;
 for(i=0;i<l;i++)
    y.append(x(i));
 return y;
}

p::list matTOlist(arma::mat x)
{
 int row=x.n_rows;
 int col=x.n_cols;
 int i;
 int j;
 p::list y;
 for(i=0;i<row;i++)
 {
   p::list z;
   for(j=0;j<col;j++)
     z.append(x(i,j));
   y.append(z);
 }
 return y;
}

p::list prova1(p::list p)
{
  arma::vec v=listTOvec(p);
  return vecTOlist(v);
}
p::list prova2(p::list p)
{
  arma::mat v=listTOmat(p);
  return matTOlist(v);
}


class Mixt {
  private:
   arma::mat x=arma::zeros(2,2);
   arma::vec ind=arma::zeros(2);  // The class
  public:           // Access specifier
    Mixt(arma::mat x1, arma::vec ind1) {     // Constructor
      x=x1;
      ind=ind1;
    }
    arma::mat get_x()
    {
     return x;
    }
        arma::vec get_ind()
    {
     return ind;
    }
};

class Bin {
  private:
   arma::vec x=arma::zeros(2000);
   arma::mat grid=arma::zeros(2000,2);  // The class
  public:           // Access specifier
    Bin(arma::vec x1, arma::mat grid1) {     // Constructor
      x=x1;
      grid=grid1;
    }
    arma::vec get_bin()
    {
     return x;
    }
        arma::mat get_grid()
    {
     return grid;
    }
};

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



class EmBin {
  private:
   std::list<Bin> bin;
   arma::vec pi=arma::zeros(2);
   arma::mat mu=arma::zeros(2,2);
   arma::mat v=arma::zeros(2,2);
   arma::vec logli=arma::zeros(2); // The class
  public:           // Access specifier
    EmBin(std::list<Bin> bin1,arma::vec pi1,arma::mat mu1, arma::mat v1,arma::vec logli1) {     // Constructor
      bin=bin1;
      mu=mu1;
      pi=pi1;
      v=v1;
      logli=logli1;
    }
    std::list<Bin> get_bin()
    {
     return bin;
    }
    arma::vec get_pi()
    {
     return pi;
    }
        arma::mat get_mu()
    {
     return mu;
    }
    arma::mat get_v()
{
 return v;
}
arma::vec get_logli()
{
return logli;
}
};
class EmBinUniv
{
  private:
   arma::vec pi_u=arma::zeros(2);
   arma::vec mu_u=arma::zeros(2);
   arma::vec v_u=arma::zeros(2);
   arma::vec logli_u=arma::zeros(2);
   std::list<Bin> bin_l;
   Bin bin_u=bin_l.front();
  public:           // Access specifier
    EmBinUniv(Bin bin1,arma::vec pi1,arma::vec mu1, arma::vec v1,arma::vec logli1)
    {
      bin_u=bin1;
      mu_u=mu1;
      pi_u=pi1;
      v_u=v1;
      logli_u=logli1;
    }
    Bin get_bin_u()
    {
     return bin_u;
    }
    arma::vec get_pi_u()
    {
     return pi_u;
    }
        arma::vec get_mu_u()
    {
     return mu_u;
    }
    arma::vec get_v_u()
{
 return v_u;
}
arma::vec get_logli_u()
{
return logli_u;
}
};

// [[Rcpp::export]]};
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
class EmDim {
  private:
   arma::vec pi=arma::zeros(2);
   arma::vec mu=arma::zeros(2);
   arma::vec v=arma::zeros(2);
  public:           // Access specifier
    EmDim(arma::vec pi1,arma::vec mu1, arma::vec v1) {     // Constructor
      mu=mu1;
      pi=pi1;
      v=v1;
    }
    arma::vec get_pi()
    {
     return pi;
    }
        arma::vec get_mu()
    {
     return mu;
    }
    arma::vec get_v()
{
 return v;
}
};
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


EmBin binmixtC(arma::mat data, int cl, arma::vec R, int it, double eps, int nrep, int seed=1)
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

p::dict binmixt(p::list data, int cl, p::list R, int it, double eps, int nrep,int seed=1)
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
EmBin binmixtclassicC(arma::mat data, int cl, arma::vec R, int it, double eps, double eps1, int it1,int nrep, int seed=1)
{
    std::cout<<"Choosing the best initial guess..."<< std::endl;

    EmBin migliore=binmixtC(data,cl,R,it,eps,nrep,seed);
    std::cout<<"Estimation phase...";

    EmBin e=embingauscompC(migliore.get_bin(),migliore.get_pi(),
                   migliore.get_mu(),migliore.get_v(),eps1,it1);
    std::cout<<"Done."<< std::endl;
    return e;
}

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


p::dict binmixtclassic(p::list data, int cl, p::list R, int it, double eps, double eps1, int it1, int nrep, int seed=1)
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


arma::mat windowC(arma::mat data,arma::vec labels, bool sd, bool m, bool rms,int fin) {
  int len=data.n_rows;
  int dim=data.n_cols;
  int cont=0;
  int contm=0;
  int contrms=0;
  if(sd==true)
  {
  cont=cont+1;
  contm=1;
  contrms=1;
  }
  if(m==true)
  {
     cont=cont+1;
     if(contm==1)
     contrms=2;
     else
     contrms=1;
  }
  if(rms==true)
     cont=cont+1;
  int matdim=dim*cont+1;
  int matlen=len-fin+1;
  arma::mat data1(matlen,matdim);
  arma::vec a(matlen);
  int k;
  int i;
  int j;
  int dim_cont=0;
  arma::vec temp(fin);
  for (k=0;k<dim;k++)
  {
    for (i=fin-1;i<len;i++)
    {
      for(j=0;j<fin;j++)
        temp(j)=data(i-j,k);
      if(sd==true)
      data1(i - fin + 1, cont * k)=arma::stddev(temp);
      if(m==true)
      data1(i - fin + 1, contm + cont * k)=arma::mean(temp);
      if(rms==true)
      data1(i - fin + 1, contrms + cont * k)=sqrt(arma::accu(arma::pow(temp,2))/fin);
    }
  }
  for (i=fin-1;i<len;i++)
  {
    data1(i - fin+1,dim*cont)= labels(i);
  }
  return data1;
}

p::list window(p::list data, p::list labels,int fin, boost::python::object sd, boost::python::object m,boost::python::object rms)
{
  arma::mat data1=listTOmat(data);
  arma::vec labels1=listTOvec(labels);
  bool sd1=true;
  if(sd==false) sd1=false;
  bool m1=true;
  if(m==false) m1=false;
  bool rms1=true;
  if(rms==false) rms1=false;
  arma::mat data2=windowC(data1,labels1,sd1,m1,rms1,fin);
  p::list ret=matTOlist(data2);
  return ret;
}





void printavec(p::list x)
{
 arma::vec y=listTOvec(x);
 std::cout<<y<<std::endl;
 p::list z;
 z=vecTOlist(y);
 std::cout<<listTOvec(z);
}
void printamat(p::list x)
{
 arma::mat y=listTOmat(x);
 std::cout<<y<<std::endl;
 p::list z;
 z=matTOlist(y);
 std::cout<<listTOmat(z);
}






BOOST_PYTHON_MODULE(example)
{
    Py_Initialize;
    np::initialize();
    p::def("dmixt", &dmixt);
    p::def("printavec",&printavec);
    p::def("printamat",&printamat);
    p::def("genmixt",&genmixt);
    p::def("buildbin",&buildbin);
    p::def("dmixt",&dmixt);
    p::def("buildbinmarg",&buildbinmarg);
    p::def("loglimult",&loglimult);
    p::def("loglimixt",&loglimixt);
    p::def("loglimarg",&loglimarg);
    p::def("embingauscomp",&embingauscomp);
    p::def("binmixt",&binmixt);
    p::def("binmixtclassic",&binmixtclassic);
    p::def("prova1",&prova1);
    p::def("prova2",&prova2);
    p::def("window",&window);
    p::def("posterior",&posterior);
}
