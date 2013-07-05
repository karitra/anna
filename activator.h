#ifndef ANN2_ACTIVATOR_INCL
#define ANN2_ACTIVATOR_INCL


#include <boost/serialization/shared_ptr.hpp>

#include "common.h"

#define SIGMA_SCALE          0.05 //and 1.0 - bad // was 0.5/0.2 - good // 0.1-exellent!


namespace ann2 {
  template<typename T> class activation_base_t;
  typedef activation_base_t<real_type> activation_base_real_t;

  template<typename T> class sigm_af_t;
  typedef sigm_af_t<real_type> sigmoid_real_fa_t;

  typedef boost::shared_ptr<activation_base_real_t> activation_ptr_t;
}

template<typename T>
class ann2::activation_base_t {
public:
  virtual T operator()(const T &net) { throw std::invalid_argument( "Activation function not implemented!" ); }
};

template<typename T>
class ann2::sigm_af_t : public ann2::activation_base_t<T> {
  T b;
public:

  sigm_af_t(const T &bb=SIGMA_SCALE) : b(bb) {}

  virtual T operator()(const T &net) {
	return ( 1 
			 / //----------------
			(1 + exp(- b * net)));
  }
};

#endif
