#ifndef ANN2_COMMON_INC
#define ANN2_COMMON_INC

#include <boost/iterator/counting_iterator.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>


#if 0
#define dbg(m) std::cerr << m << std::endl;
#else 
#define dbg(m)
#endif

#define say(m) std::cout << m <<std:: endl;

// TODO: throw an exception
#define err(m) std::cerr << m << std::endl;


#define  lambda(t)       [ ] (t) -> void 
#define  lambda_op(r,t)   r  (t) -> void 
#define  lambda_ref(...) [__VA_ARGS__]


namespace ann2 {
  namespace ublas = boost::numeric::ublas;

  typedef boost::counting_iterator<int> iter_cnt_t;

  typedef double real_type;
  typedef real_type base_type;

  typedef ublas::matrix<real_type>          matrix_real_t;
  typedef ublas::vector<real_type>          vector_real_t;
  typedef ublas::diagonal_matrix<real_type> diag_matrix_real_t;
  
  typedef vector_real_t      * vector_real_ptr_t;
  typedef matrix_real_t      * matrix_real_ptr_t;
  typedef diag_matrix_real_t * diag_matrix_real_ptr_t;
		   
  typedef boost::shared_ptr<vector_real_t> vector_real_shptr_t;
}

#endif
