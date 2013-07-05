#ifndef ANN2_INCL
#define ANN2_INCL


#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <map>
#include <utility>

// Not needed as specialized definition taken from
// serialization library is used!
//#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/progress.hpp>

#include "common.h"
#include "generator.h"
#include "activator.h"

#define EPOCH_SATISFY_ERROR  0.000001 //0.000000005
#define NET_SATISFY_ERROR    0.00005 // was 0.001 - covergence at ~200 th iterations

#define DEFAULT_AMMORTIZATION  0.3 // 0.2 - not sure was it ok?  // to overflow possible local minimum
#define DEFAULT_LEARNING_RATE  0.7 // 0.45 - very slow // 0.8  // learn, but not so fast ;)

#define WEIGHTS_INIT_FROM  -0.05
#define WEIGHTS_INIT_TO     0.05

#define MAX_EPOCHS               1000
//#define EPOCH_LOW_WATERMARK      5
//#define GOOD_SAMPLES_PROBABILITY 0.85

#define CLUSTER_LEARNING_NORM     0.9
#define CLUSTER_LEARNING_SUBNORM  0.1


namespace ann2 {
  enum ClusterProcType {
	Euclidian,
	Dotproduct
  };
  
  class layer_t;
  typedef layer_t * layer_ptr_t;

  class network_t;
  class net_mentor_base_t;
  class feed_back_t;
  class centroid_dump_t;

  template<ClusterProcType T> class cluster_net_t;

  typedef std::pair<int,int> layer_demension_t;

  void store_net(const char *fname, const ann2::network_t &net);
  void load_net(const char *fname, ann2::network_t &net);

  std::ostream &operator<<(std::ostream &o, const layer_t &l);
}


class ann2::net_mentor_base_t {
public:
  virtual ~net_mentor_base_t()   {}

  virtual void updateFeedBack()  {}
  virtual void dump_cluster_pos(size_t winner, const vector_real_t &pos) {}
};

class ann2::centroid_dump_t : public ann2::net_mentor_base_t {
  boost::progress_display progress;
  std::string fpfx;
 public:
  
 centroid_dump_t(int i, const std::string &pfx=std::string("")) : progress(i), fpfx(pfx) {}
  
  void dump_cluster_pos( size_t winner, const vector_real_t &pos) {
	dump_cluster_pos( ( fpfx.size() ) ? fpfx : "cluster_", winner, pos);
  }

  void dump_cluster_pos(size_t winner, const vector_real_t &pos, const std::string &label=std::string("")) {
	dump_cluster_pos(fpfx, winner, pos);
  }

  void dump_cluster_pos(const std::string &fname_pfx, size_t winner, const vector_real_t &pos, const std::string &label=std::string("")) {
	using namespace std;

	ostringstream os;
	os << fname_pfx << winner << ".dat";

	++progress;

	dbg("Writing centroid to file " << os.str() );
	ofstream of(os.str().c_str(), ios_base::app);

	if (!of)
	  return;

	dbg("Writing...");
	//	copy(pos.begin(), pos.end(), std::ostream_iterator<real_type&>(of, " "));
	for_each(pos.begin(), pos.end(), 
			 [&of] (const real_type &v) { of << v << ' '; } );
	of << label << endl;

	of.close();
  }

};

class ann2::feed_back_t : public ann2::net_mentor_base_t {
  boost::progress_display progress;
public:
  feed_back_t(int l) : progress(l) {}

  void updateFeedBack() {
	++progress;
  }
};


namespace boost {
  namespace serialization {

	template<class Archive>
	void save(Archive &ar, const ann2::matrix_real_t &m, const unsigned int version)
	{
	  dbg("Saving matrix_real_t: " << m);
	  std::ostringstream os;
	  os << m;
	  std::string ms = os.str();
	  ar << ms;
	}

	template<class Archive>
	void load(Archive &ar, ann2::matrix_real_t &m, const unsigned int version)
	{
	  dbg("Restoring matrix_real_t");
	  std::string s;
	  ar >> s;
	  std::stringstream ss;
	  ss << s;
	  ss >> m;

	  dbg("Get matrix: " << m);
	}

	
	template<class Archive>
	void serialize(Archive &ar, ann2::matrix_real_t &m, const unsigned int version)
	{
	  dbg("Serializing matrix_real_t");
	  //split_free(ar, m, version );
	  //	  ar & m;
	  m.serialize(ar, version );
	}
  }

}

// @brief network layer 
//
// TODO: make separation of the states for study and working mode, to reduce the size
// and number of matrixes needed while working in 'normal' mode.
//
class ann2::layer_t {
  friend class ann2::network_t;

  friend class boost::serialization::access;
  template<class Archive> 
  void serialize(Archive &ar, const unsigned int version) {
	  ar & in;
	  ar & out;

	  ar & W;
  }

  boost::shared_ptr<matrix_real_t> W; // must be stored
  boost::shared_ptr<matrix_real_t> dWprev;

  boost::shared_ptr<vector_real_t> Ain;

  boost::shared_ptr<matrix_real_t> D;
  boost::shared_ptr<diag_matrix_real_t> In;

  int in, out; // must be stored

  // Used for 'back propogation'
  layer_t *next_layer;

  activation_ptr_t activate_func; // should be stored, but I don't know how

public:

  layer_t(int i, int o, layer_t *nxt=0, const activation_ptr_t &ap=activation_ptr_t(new sigmoid_real_fa_t)) : 
	W(     new matrix_real_t(i, o)),
	in(i), out(o), 
	next_layer(nxt),
	// We don't need to init them in normal proc,
	//  only when studying!
	// dWprev(new matrix_real_t(i, o)),
	// Ain(   new vector_real_t(i)),
	// D(     new matrix_real_t(i, o) ),
	// In(    new diag_matrix_real_t(i, i) ),
	activate_func( ap )
  {	  
	init();
  }

  int getOutUnitsNum() const {
	return out;
  }

  int getInUnitsNum() const {
	return in;
  }

  layer_demension_t get_demension() const {
	return std::make_pair(in - 1, out - 1);
  }

  layer_t *setNext(layer_t *next) {
	return next_layer = next;
  }

  void setInMatrix(const vector_real_t &v) {
	for(unsigned int i = 0; i < In->size1(); i++)
	  for(unsigned int j = 0; j < In->size2(); j++) 
		if (i == j) {
		  (*In)(i,j) = v(i);
		  (*Ain)(i)  = v(i); // need to set it for previous leayer (if it exist)
		}
  }

  matrix_real_ptr_t getW()   { return W.get();   }
  matrix_real_ptr_t getD()   { return D.get();   }
  vector_real_ptr_t getAin() { return Ain.get(); }

  activation_base_real_t *getActivation() const {
	return activate_func.get();
  }

  // Output layer weight update
  void updateWeights(const vector_real_t &d, const vector_real_t &y, 
					 const double &alpha=double(DEFAULT_LEARNING_RATE), const double &mu=(DEFAULT_AMMORTIZATION) );
  void updateWeights( const double &alpha=double(DEFAULT_LEARNING_RATE), const double &mu=(DEFAULT_AMMORTIZATION) );

  friend std::ostream &operator << (std::ostream &o, const layer_t &l);  

private:

  void init_study_arrays() {
	dWprev.reset ( new matrix_real_t(in, out)      );
	Ain.reset    ( new vector_real_t(in)           );
	D.reset      ( new matrix_real_t(in, out)      );
	In.reset     ( new diag_matrix_real_t(in, in)  );

	std::for_each(iter_cnt_t(0),
		 iter_cnt_t(dWprev->size1()),
		 [&dWprev] (int i) -> void {
					ublas::matrix_row<matrix_real_t> r(*dWprev, i);
					fill(r.begin(), r.end(), 0.0);
		 } );

	dbg("dWprev = " << *dWprev);
  }

  void delete_study_arrays() {
	dWprev.reset ();
	Ain.reset    ();
	D.reset      ();
	In.reset     ();
  }

  void init() {

	if (!activate_func) 
	  activate_func.reset( new sigmoid_real_fa_t );

	// Those one have to be initialized
	init_W();
	//init_I();
  }

  void init_W() {
	using namespace std;
	using namespace boost;
	using namespace ublas;

	matrix_real_t &m = *W;
	rnd_real_gen_t rnd( WEIGHTS_INIT_FROM, WEIGHTS_INIT_TO );

	for_each(iter_cnt_t( 0 ),
			 iter_cnt_t( m.size1() ),
			 lambda_op( lambda_ref(&m, &rnd), int i) {
			   matrix_row< matrix_real_t > r(m, i);
			   generate( r.begin(), r.end(), rnd);
			   //fill( r.begin(), r.end(), 0.0);
			 } );

  }

};


namespace boost {
  namespace serialization {
	
	template<class Archive>
	inline void save_construct_data(Archive & ar, const ann2::layer_t * l, const unsigned int version ) {
	  int 
		in  = l->getInUnitsNum(),
		out = l->getOutUnitsNum();

	  ar << in;
	  ar << out;
	}

	template<class Archive>
	inline void load_construct_data( Archive & ar, ann2::layer_t * t, const unsigned int version) {
	  int in, out;

	  ar >> in;
	  ar >> out;

	  ::new(t) ann2::layer_t(in, out);
	}

  }
}


class ann2::network_t {

  friend class boost::serialization::access;
  template<class Archive> 
  void serialize(Archive &ar, const unsigned int version) {
	dbg("Serializing network");
	if (version)
	  ar & life_cycle_epochs;

   	ar & layers;
	if (Archive::is_loading::value)
	  restore_next_links();
  }


  typedef boost::shared_ptr<layer_t> layer_smart_ptr_t;
  typedef std::vector<layer_smart_ptr_t> layers_vector_t; 
  layers_vector_t layers;

  // Study proccess variables
  real_type total_sqr_err, prev_epoch_err;

  // Note: deprecated!
  unsigned int good_samples_count;

  int life_cycle_epochs;

public:

  network_t() : life_cycle_epochs(0) {}

  network_t *add_layer(int in_units, int out_units, 
					   bool biased=true, 
					   activation_base_real_t *af=0) throw(std::invalid_argument );

  
  int getTotalEpochs() const { return life_cycle_epochs; }

  layer_demension_t get_demension(int i) const {
	return layers.at(i)->get_demension();
  }

  int layers_number() const 
  {
	return layers.size();
  }

  /* Unused! Remove it

  // @param   y real output
  // @param   d desired output
  // @return is error accessable
  static bool satisfaction(const vector_real_t &d, const vector_real_t &y, 
						   const double &err=double(EPOCH_SATISFY_ERROR)) {
	// Satisfaction condition:
	// e < err, where
	// e = sum| diff * diff^t| 
	// We can do here without the norm, but is look more pedantic to apply it here,
	// as prod still returns (one element) vector, and we need to take first element from
	// it which is not as beautiful as following solution:
	return ( sample_error(d, y) / 2.0  < err / 2.0) ? true : false;
	}
  */

  static real_type sample_error(const vector_real_t &d, const vector_real_t &y);

  // Returns number of epoch taken
  int study(base_sample_gen_t &sgen,
			const int epoch_limit=int(MAX_EPOCHS),
			net_mentor_base_t *vizier=0,
			const double &epsilon=double(EPOCH_SATISFY_ERROR),
			const double &net_err_limit=double(NET_SATISFY_ERROR) );



	// Note: biased version
	vector_real_t proc(const vector_real_t &acc,
					   net_mentor_base_t *vizier=0) {
	return std::accumulate(layers.begin(), 
						   layers.end(), 
						   acc, 
						   [&vizier] (const vector_real_t &in, const layer_smart_ptr_t &lw) 
						   { 
						   	 vector_real_t out = ublas::prod(in, *lw->getW() ); 
						   	 std::transform(out.begin()+1, out.end(), out.begin()+1, 
						   					functor_t<real_type>(lw->getActivation()) );
						   	 out(0) = 1.0;

						   	 if (vizier)
						   	   vizier->updateFeedBack();

						   	 return out;
						   } );
  }

private:

  void init_study_vars() {
    good_samples_count =   0;
    total_sqr_err      = 0.0;
	prev_epoch_err     = 0.0;

	std::for_each(layers.begin(), 
				  layers.end(),
				  [] (layer_smart_ptr_t &sp) { 
					sp->init_study_arrays();
				  }
				  );
  }

  // Used after serialization process has been completed
  void restore_next_links() {
	for(layers_vector_t::iterator i = layers.begin();
		i != layers.end(); i++ ) {
	  if ((i+1) != layers.end())
		i->get()->setNext( (i + 1)->get() );
	}
  }

  void delete_study_vars() {
	std::for_each(layers.begin(), 
				  layers.end(),
				  [] (layer_smart_ptr_t &sp) { 
					sp->delete_study_arrays();
				  }
				  );
  }

  bool stop_condition(const vector_real_t &d,
					  const vector_real_t &y,
					  unsigned int &epoch,
					  const unsigned int epoch_limit,
					  const real_type &epoch_err,
					  const real_type &epoch_err_limit,
					  const real_type &net_err_limit);

  

  template<class T>
  struct functor_t {
	activation_base_t<T> *af;
	functor_t(activation_base_t<T> *a) : af(a) {}

	T operator()(const T &net) {
	  return (*af)(net);
	}
  };


  vector_real_t proc_study(const vector_real_t &acc) {
	return std::accumulate(layers.begin(), layers.end(), 
						   acc, 
						   [] (const vector_real_t &in, layer_smart_ptr_t &lw) 
						   { 
							 
							 vector_real_t net = ublas::prod(in, *lw->getW() ); 
							 /*
							 dbg("in: " << in);
							 dbg("W" <<  *lw->getW() );
							 dbg("net: " << net);
							 */

							 std::transform(net.begin()+1, net.end(), net.begin() + 1, 
											functor_t<real_type>(lw->getActivation()) );
							 // dbg("out: " << net);
							 net(0) = 1.0;

							 lw->setInMatrix(in); // story input for study process

							 return net;
						   } );
  }

};


BOOST_CLASS_VERSION(ann2::network_t, 1)


template<ann2::ClusterProcType Type=ann2::ClusterProcType::Euclidian>
class ann2::cluster_net_t {
  struct lnorm_base_t {
	virtual real_type lrn1(int e) const = 0;
	virtual real_type lrn2(int e) const = 0;
  };

  struct lnorm_conv_t : public lnorm_base_t {
	real_type lrn1(const int epoch) const {
	  return 0.94 * exp(-epoch/1000) + 0.01;
	}

	real_type lrn2(const int epoch) const {
	  return 0.1499 * exp(-epoch/300) + 0.0001;
	}

	static lnorm_base_t *create() {
	  return new lnorm_conv_t;
	}
  };

  struct lnorm_attract_t : public lnorm_base_t {
	real_type lrn1(const int epoch) const {
	  return 0.94 * exp(-epoch/2000) + 0.01;
	}

	real_type lrn2(const int epoch) const {
	  return 0.2 * exp(-epoch/500) + 0.007;
	}

	static lnorm_base_t *create() {
	  return new lnorm_attract_t;
	}
  };

private:
  int in, out;

  boost::shared_ptr<matrix_real_t> W; // must be stored
  boost::shared_ptr<vector_real_t> y; 

  boost::shared_ptr<lnorm_base_t> lnorm;

  int epoch;

  // Learing norms
  real_type lnorm1, lnorm2;
public:

  cluster_net_t() : 
	in(0), out(0),
	W(new matrix_real_t),
	y(new vector_real_t)
  { init(); }
  
  cluster_net_t(int i, int o) :
	in(i), out(o),
	W(new matrix_real_t(in, out)),
	y(new vector_real_t(out))
  { init(); }

  cluster_net_t(int o, const cluster_bounds_vec_t &bv, bool is_conv=true) :
	in(bv.size()), out(o),
	W(new matrix_real_t(in, out)),
	y(new vector_real_t(out))
  { init(&bv); }


  void init(const cluster_bounds_vec_t *const bounds=0, bool is_conv=true) {
	lnorm1 = CLUSTER_LEARNING_NORM;
	lnorm2 = CLUSTER_LEARNING_SUBNORM;
	epoch  = 0;

	// W already created so adjust it weights
	initW(bounds);

	lnorm.reset(is_conv ? lnorm_conv_t::create() : lnorm_attract_t::create() );
  }


  void set_inputs_num(int n) {
	reset_weight_matrix(n, out);
  }

  void add_cluster_unit() {
	reset_weight_matrix(in, out + 1);
  }

  size_t get_clusters_num() {
	return out;
  }

  // TODO: optimize returning of the vector someway
  vector_real_t get_cluster_centre(size_t j) {
	return ublas::matrix_column<matrix_real_t>( *W, j );
  }

  vector_real_t operator[](size_t j) {
	dbg("Requested centre " << j << " number of centres " << W->size2());
	return get_cluster_centre(j);
  }

  size_t clusters_number() const {
	return W->size2();
  }

  size_t proc(const vector_real_t &input);

  void run(base_cluster_gen_t &gen, const int epoch_limit, net_mentor_base_t *v=0) {
	using namespace std;

	for(int ep = 0;  ep < epoch_limit; ep++, epoch++) {

	  for_each(iter_cnt_t( 0 ),
			   iter_cnt_t( gen.size() ),
			   [&gen, v, this] (int i) -> void {
				 
				 dbg("Iteration #" << i);
				 vector_real_t *sample = gen[i];
				 if (not sample)
				   return;

				 //				 size_t winner = this->proc(*sample);
				 this->proc(*sample);

		   		 dbg("Dumping of centroid position!");
			   } );

	  if (v)
		for_each(iter_cnt_t(0), iter_cnt_t(clusters_number()),
				 [this,&v] (int i) {
				   v->dump_cluster_pos(i, (*this)[i] ); } );
	   	  
	  gen.shuffle();
	}

  }


  // Norms properties
  real_type get_lnorm1() const {
	return lnorm1;
  }
  
  void set_lnorm1(const real_type &val) {
	lnorm1 = val;
  }

  real_type get_lnorm2() const {
	return lnorm2;
  }
  
  void set_lnorm2(const real_type &val) {
	lnorm2 = val;
  }


  std::ostream &dump_weights(std::ostream &o) const {
	using namespace std;

	for_each( iter_cnt_t(0),
			  iter_cnt_t(W->size2()),
			  [&o, &W] (int j) {
			    for_each(iter_cnt_t(0),
						 iter_cnt_t(W->size1()),
						 [&o,&j,&W] (int i) {
						   o << W->operator()(i,j) << ' ';
						 });
			  });

	return o;
  }


private:
  void   initW(const cluster_bounds_vec_t *const bounds);
  
  size_t update_winner(const vector_real_t &x, const size_t winner_i, bool normalize=false) {
	using namespace ublas;

	const int ep = epoch;
	std::for_each(iter_cnt_t(0), 
				  iter_cnt_t(W->size2() ),
				  [&x, &W, this, &ep, normalize, &winner_i] ( int i ) -> void {
					matrix_column<ann2::matrix_real_t> output(*W, i);
					output.plus_assign( (( (size_t ) i == winner_i ) ? this->lnorm->lrn1(ep) : this->lnorm->lrn2(ep)) * (x - output) );
					if (normalize)
					  output *= 1 / norm_2(output);
				  } );

	return winner_i;
  }

  void reset_weight_matrix(int i, int o) {
	in = i, out = o;
	W->resize(in, out); // preserve = true
	// todo: reset inputs
  }


};

namespace ann2 {
  template<ClusterProcType T> std::ostream &operator<<(std::ostream &o, const cluster_net_t<T> &n ) {
	return n.dump_weights(o);
  }
}

#endif
