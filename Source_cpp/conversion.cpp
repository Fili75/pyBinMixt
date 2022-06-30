#include "conversion.h"
#include <boost/python/extract.hpp>
#include <vector>

namespace p = boost::python;


arma::vec listTOvec(p::list x)
{
 int l=p::len(x);
 int i;
 arma::vec y(l);
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