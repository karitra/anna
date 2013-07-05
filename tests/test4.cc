#include <algorithm>
#include <iterator>
#include <vector>
#include <iostream>
#include <numeric>

#include <ctime>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/progress.hpp>

class dummy_t;
using namespace std;
namespace ublas = boost::numeric::ublas;
typedef boost::shared_ptr<dummy_t> dummy_ptr_t;


class dummy_t {
  static int cnt;
  int val;
public:
  dummy_t() : val(++cnt) {}
  dummy_t(int v) : val(v) {}

  int get_some() const{ return val; }

  const dummy_t operator+(const dummy_t &op2) {
	return dummy_t(op2.get_some() + val);
  }

  operator int () {
	return val;
  }

   friend dummy_t     operator+(const dummy_t &op1, const dummy_t &op2);
   friend dummy_ptr_t operator+(const dummy_ptr_t &a, const dummy_ptr_t &b);
   //  friend template<dummy_ptr_t>  boost::shared_ptr<dummy_t> operator+(const boost::shared_ptr<dummy_t> &a, const boost::shared_ptr<dummy_t> &b);
 };

 dummy_t operator+(const dummy_t &op1, const dummy_t &op2)
 {
   return dummy_t( op1.val + op2.val);
 }

 dummy_ptr_t operator+(const dummy_ptr_t &a, const dummy_ptr_t &b) {
   return dummy_ptr_t(new dummy_t(a->get_some() + b->get_some() ) );
 }

 template<class T>
 boost::shared_ptr<T> operator+(const boost::shared_ptr<T> &a, const boost::shared_ptr<T> &b)
 {
   return boost::shared_ptr<T>(new T(*a.get() + *b.get()));
 }



 int dummy_t::cnt = 0;

 #define MAX_LIMIT 10000000

 struct int_inc_t {
   int i;
   int_inc_t(int ii = 0) : i(ii) {}

   int operator()() { return ++i; }

 };

 template<class T1, class T2>
 struct op2_t {

   T1 operator()(const T1 &acc, const T2 &operand) {
	 return T1(10);
   }

   T1 operator()(const T1 &acc, const T1 &operand) {
	 return T1(10) + static_cast<T1>(operand);
   }
 };

template<class Guest>
class holder_t : public boost::shared_ptr<Guest> {
public:
  holder_t(Guest *p) : boost::shared_ptr<Guest>(p) {}
  operator int () {
	return 10;
  }
};

 int
 main(int argc, char *argv[])
 {
   typedef std::vector<holder_t<dummy_t> > holder_arr_t; 
   vector<double> a;
   ublas::vector<double> v(10000);
   holder_arr_t dp;

   clock_t start = clock();
   generate_n( back_inserter(a), MAX_LIMIT, int_inc_t() );
   generate( v.begin(), v.end(), int_inc_t() );
   fill_n( back_inserter(dp), MAX_LIMIT, holder_t<dummy_t>(new dummy_t) );

   int i = 5;
   double summ = 0.0;
   boost::progress_display progress(i);
   while(i--) {
	 dummy_t dm(0);

	 sort(a.begin(),   a.end() );
	 sort(v.begin(),   v.end() );
	 sort(dp.begin(), dp.end() );

	 summ =  accumulate(a.begin(), a.end(), summ );
	 summ =+ accumulate(v.begin(), v.end(), summ );


	 int acc;
	 //	 std::vector<dummy_ptr_t>::iterator start = dp.begin();
	 //std::vector<dummy_ptr_t>::iterator end   = dp.end();
	 acc  =  accumulate( dp.begin(), dp.end(), acc, op2_t<int, holder_t<dummy_t> >() );
	 ++progress;
  }

  clock_t end   = clock();

  cerr << "Time taken: " << ((double) (end - start) / CLOCKS_PER_SEC) << " sec!"<< endl;
  cerr << "Summ: " << fixed << summ << endl;
  return 0;
}
 
