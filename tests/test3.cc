#include <algorithm>
#include <iostream>
#include <typeinfo>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/iterator/counting_iterator.hpp>

int
main(int argc, char *argv[])
{
  using namespace boost::numeric::ublas;
  using namespace std;

  matrix<double> m(3, 2);
  //  cerr << typeid(m).name() << endl;
  int    i = 0;
  double d = 0;

  matrix<double>::iterator1 it = m.begin1();

  cerr << typeid(double).name() << endl 
	   << m(1,1) << endl;

  cerr << typeid(it).name() << endl;

  int val = 0;
  for(int i = 0; i < m.size1(); i++)
	for(int j =0; j < m.size2(); j++)
	  m(i,j) = ++val;

  cerr << m << endl;
  
  for_each(m.begin1(), m.end1(),
		   [] (double &v) -> void 
		   {cerr << "v: " << v  << endl;} );

  using namespace boost;
  for_each(counting_iterator<int>(0),
		   counting_iterator<int>(m.size1()),
		   [&m] (int i) -> void {

			 matrix_row< matrix<double> > mr(m, i);
			 for_each(mr.begin(), mr.end(), 
			  		  [] (double &v) -> void 
			  		  {cerr <<"v = " << v << endl;} );
		   } );

  range r(0, m.size1() );
  range c(0, m.size2() );

  for(uint32_t i = 0; i < m.size1(); i++)
	for(uint32_t j = 0; j < m.size2(); j++)
	  cerr << "m: " << m(i, j) << endl;

  for_each(r.begin(), r.end(),
		   [&c, &m] (unsigned int i) -> void {
			 for_each(c.begin(), c.end(),
					  [&i, &m] (int j) -> void
					  {
						cerr << "M = " << m(i, j) << endl;
					  } );
		   } );

  return 0;
}
 
