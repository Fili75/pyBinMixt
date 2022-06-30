#include "conversion.h"
#include "window.h"

namespace p=boost::python;

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