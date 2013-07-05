#ifndef ANN2_GEN_INCL
#define ANN2_GEN_INCL

#include <ctime>

#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

#include  "common.h"

#define SAMPLE_POINTS_NUMBER     10
#define PARABOLIC_POINTS_NUMBER 100
#define CLUSTER_SAMPLES_NUM      32

namespace ann2 {
  typedef std::pair<real_type, real_type>   dims_bounds_t;
  typedef std::vector<dims_bounds_t>        cluster_bounds_vec_t;
  typedef std::vector<cluster_bounds_vec_t> clusters_domains_t;

  typedef std::pair<real_type, real_type> sample_t;

  class gen_container_t;
  class base_sample_gen_t;
  class base_cluster_gen_t;
  template<int Num> class cluster_samples_gen_t;
  template<int Num> class sample_clusters_db_t;


  template <class T, unsigned int N> class sinus_generator_t;
  template <unsigned int N>          class parabolic_generator_t;

  template<class T> class scale_to_one_t;
  template<class T> class scale_back_t;

  struct approx_rect;

  class rnd_real_gen_t;

  template<int N>
  std::ostream &operator << (std::ostream &os, const cluster_samples_gen_t<N> &smp)
  {
	return smp.dump_samples(os);
  }

}


class ann2::rnd_real_gen_t {
  boost::mt19937 gen;
  boost::uniform_real<> dist;
  boost::variate_generator<boost::mt19937 &, boost::uniform_real<> > rnd;

  static unsigned long rand_seed;

public:

  rnd_real_gen_t(const real_type &from, const real_type &to, 
				 unsigned int seed = 0) : 
	//	gen(seed = rand_seed = ( (rand_seed ^ 0xA5A5) << (rand_seed & 0x3)) + rand_seed++)), 
	gen(seed),
    dist(from, to), rnd(gen, dist) {
	seed++;
	rand_seed += ((seed ^ 0xA5A5) << (seed & 0x3)) + seed;
	gen.seed( rand_seed );
  }

	double operator()() { return rnd(); }
};


struct ann2::approx_rect {
  real_type x1, x2, y1, y2;

  approx_rect(const real_type &x_from,
			  const real_type &x_to,
			  const real_type &y_from,
			  const real_type &y_to ) : 
	x1(x_from), x2(x_to),
	y1(y_from), y2(y_to) {}
};

template<class T> 
class ann2::scale_back_t {
  T from, to;
public:
  scale_back_t(const T &f, const T &t) : from(f), to(t) {}
  
  const T operator()(const T &v) const { return v * (to - from) + from; }
};

// Scales value to square with side size = 1
template<class T>
class ann2::scale_to_one_t {
 T from, to;
public: 
  scale_to_one_t(const T &f, const T &t) : from(f), to(t) {}
  
  const T operator()(const T &v) const { 
	return
	  (v - from) 
	  / //-------------------------------------
	  (to - from); 
  }
};


class ann2::gen_container_t {
public:
  virtual size_t size() const = 0;
  virtual void shuffle()      = 0;
};

class ann2::base_sample_gen_t : public virtual ann2::gen_container_t {
public:
  virtual vector_real_t  input(size_t) const = 0;
  virtual vector_real_t output(size_t) const = 0;
};

class ann2::base_cluster_gen_t : public virtual ann2::gen_container_t {
protected:
  std::vector<vector_real_shptr_t> samples;

public:

  virtual size_t  size() const { return samples.size(); }
  virtual void shuffle() { std::random_shuffle(samples.begin(), samples.end() ); }

  // Note: unsafe, update to more sequre version someday!
  virtual vector_real_t *operator[](int i) {
	dbg("Generator: requested number " << i << " in " <<  samples.size() );
	return samples[i].get();
  }


  virtual std::ostream &dump_samples(std::ostream &os) const {
	using namespace std;
	
	for_each( samples.begin(),
			  samples.end  (),
			  [&os] (const vector_real_shptr_t &vp) -> void {
				copy( vp->begin(), vp->end(), std::ostream_iterator<real_type&>(os, " ") );
				os << endl;
			  });	

	return os;
  }

};


template<class T, unsigned int N=SAMPLE_POINTS_NUMBER>
class ann2::sinus_generator_t : public ann2::base_sample_gen_t {
  T x1, x2, y1, y2;
  scale_to_one_t<real_type> xscale, yscale;
  scale_back_t<real_type> xzoom, yzoom;
public:

  sinus_generator_t(const approx_rect &arec) :
	x1(arec.x1), x2(arec.x2),
	y1(arec.y1), y2(arec.y2),
	xscale(arec.x1,arec.x2), 
	yscale(arec.y1,arec.y2),
	xzoom(arec.x1,arec.x2),
	yzoom(arec.y1,arec.y2)
  { 
	using namespace std;
	double x, y, dx = (x2 - x1) / N;

	std::for_each(boost::counting_iterator<int>(0),
				  boost::counting_iterator<int>(N),
				  [&x, &y, &dx, &x1,&x2,&samples,&xscale,&yscale] (int i) -> void {
					x = x1 + dx * i + dx / 2.0;
					y = sin(x);
					samples.push_back(make_pair( xscale(x), yscale(y) ) );
				  } );
  }

  size_t size() const {return N;}

  void shuffle() { std::random_shuffle(samples.begin(), samples.end() ); }

  // Note: input, output vectors are biased!
  vector_real_t  input(size_t i) const {
	vector_real_t in(2);
	in(0) = 1.0;
	in(1) = samples[i].first;
	
	return in;
  }

  vector_real_t output(size_t i) const {
	vector_real_t out(2);
	out(0) = 1.0;
	out(1) = samples[i].second;

	return out;
  }

  // Dump samples to file 'fname' in gnuplot aware format
  // TODO: 
  // - move it out of header;
  // - make streams version;
  void dump_samples(const char *fname=0)
  {
	using namespace std;
	
	ostringstream os;

	if (fname) {
	  os << fname;
	} else {
	  os << "samples_scaled_" << samples.size() << ".dat";
	}
	
	ofstream sf( os.str().c_str(), ios_base::app);
	if (!sf) {
	  err("Failed to open file" << os << '!');
	  return;
	}

	for(size_t i = 0;i < samples.size(); i++) {
	  sample_t s = samples[i];
	  sf << s.first << ' ' << s.second << endl;
	}

	sf.close();
  }

  void dump_original_samples(const char *fname=0)
  {
	using namespace std;
	
	ostringstream os;

	if (fname) {
	  os << fname;
	} else {
	  os << "samples_orig_" << samples.size() << ".dat";
	}
	
	ofstream sf( os.str().c_str(), ios_base::app);
	if (!sf) {
	  err("Failed to open file" << os << '!');
	  return;
	}

	for(size_t i = 0; i < samples.size(); i++) {
	  sample_t s = samples[i];
	  sf << xzoom(s.first) << ' ' << yzoom(s.second) << endl;
	}

	sf.close();
  }

private:
  virtual const sample_t &operator[](const int i) const { return samples[i]; }

private:
  typedef std::vector<sample_t> samples_vector_t;
  samples_vector_t samples;
};


namespace std {
  template<>
  // FIXME: not working!
  inline void swap<ann2::vector_real_shptr_t>(ann2::vector_real_shptr_t &a, ann2::vector_real_shptr_t &b) {
	dbg("Vector swapping!");
   	a.swap(b);
  }
}



template<unsigned int N=PARABOLIC_POINTS_NUMBER>
class ann2::parabolic_generator_t : 
  public ann2::base_cluster_gen_t,
  public ann2::base_sample_gen_t
{
public:

  //  parabolic_generator_t(std::initializer_list<cluster_bounds_vec_t> list) {
  //	gen_samples(list);
  //}

  parabolic_generator_t(const cluster_bounds_vec_t &bvec) {
	gen_samples(bvec);
  }

  // Note: input, output vectors are biased!
  vector_real_t  input(size_t i) const {
  	vector_real_t in(2);
  	in(0) = 1.0;
  	in(1) = (*samples[i])(0);
	
  	return in;
  }

  vector_real_t output(size_t i) const {
  	vector_real_t out(2);
  	out(0) = 1.0;
  	out(1) = (*samples[i])(1);

  	return out;
  }


  void gen_samples(const cluster_bounds_vec_t &bvec) {
	using namespace std;
	using namespace ublas;
	enum Coord {X, A};

	samples.clear();

	if (bvec.size() != 2) // not a Decart mapping
	  throw std::runtime_error(string(__func__) + " incorrect init vector size for this generator");

	real_type x,y,dx;
	dx = (bvec[X].second - bvec[X].first) / N;

	dbg("x1: " << bvec[X].first  );
	dbg("x2: " << bvec[X].second );

	dbg("a: " << bvec[A].first  );
	dbg("b: " << bvec[A].second );


	for_each(iter_cnt_t(0),
			 iter_cnt_t(N),
			 [&x, &y, &dx, &samples, &bvec] (int i) -> void {

			   x  = bvec[X].first + i * dx +  bvec[A].first;
			   y  = x * x + bvec[A].second;

			   vector_real_shptr_t pv(new vector_real_t( bvec.size() ) );
			   (*pv)(0) = x -  bvec[A].first;
			   (*pv)(1) = y;
			   samples.push_back(pv);
			 } );
  }


};



template<int Num=CLUSTER_SAMPLES_NUM>
class ann2::cluster_samples_gen_t : public ann2::base_cluster_gen_t {
  //  std::vector<vector_real_shptr_t> samples;
public:
  
  cluster_samples_gen_t(const cluster_bounds_vec_t &bvec) {
	gen_samples(bvec);
  }

  cluster_samples_gen_t() {
	cluster_bounds_vec_t b;

	// Default: add to demensions to samples space
	cluster_samples_gen_t<Num>::add_dim_bound(b, 0.0, 1.0);
	cluster_samples_gen_t<Num>::add_dim_bound(b, 0.0, 1.0);

	gen_samples(b);
	//	dbg("dimensions vector size: " << b.size() );
  }


  static void add_dim_bound(cluster_bounds_vec_t &b, const real_type &from, const  real_type &to) {
	b.push_back( std::make_pair( from, to) );
  }


  void gen_samples(const cluster_bounds_vec_t &bvec) {
	using namespace std;
	using namespace ublas;

	samples.clear();

	//	dbg("In " << __func__ );
	for_each(iter_cnt_t(0),
			 iter_cnt_t(Num),
			 [&samples, &bvec] (int i) -> void {

			   // dbg("Sample #" << i);
			   vector_real_shptr_t pv(new vector_real_t(bvec.size()));
			   // dbg("pv size: " << pv->size() );
			   for_each(iter_cnt_t(0),
						iter_cnt_t(pv->size()),
						[&pv, &bvec] (int j) -> void {
						  rnd_real_gen_t rnd(bvec[j].first, bvec[j].second);
						  (*pv.get())(j) = rnd();
						} );
			   
			   samples.push_back(pv);
			 } );

	//	dbg("samples size is " << samples.size() );
  }

			 
  std::ostream &dump_samples(std::ostream &os, bool normalized=true) const {
	using namespace std;
	
	for_each( samples.begin(),
			  samples.end  (),
			  [&os, &normalized] (const vector_real_shptr_t &vp) -> void {
				vector_real_t tmp = (!normalized) ? 
				  *vp : 
				  *vp * (1/norm_2(*vp));

				copy( tmp.begin(), tmp.end(), std::ostream_iterator<real_type&>(os, " ") );
				os << endl;
			  });	

	return os;
  }

};


template<int Samples>
class ann2::sample_clusters_db_t : public ann2::base_cluster_gen_t {
  std::vector<ann2::cluster_samples_gen_t<Samples> > clusters;
  boost::mt19937 random_gen;
public:

  sample_clusters_db_t(std::initializer_list<ann2::cluster_bounds_vec_t> list) {
	init_random();
	build_clusters(list);
  }

  void build_clusters(std::initializer_list<ann2::cluster_bounds_vec_t> list) {
	init_random();
	copy( list.begin(), list.end(), back_inserter(clusters) );
  }

  void build_clusters(clusters_domains_t &dom)  {
	init_random();
	copy( dom.begin(), dom.end(), back_inserter(clusters) );
  }


  size_t clusters_number() {
	return clusters.size();
  }

  std::ostream &dump_clusters(std::ostream &o, bool normalize ) {
	using namespace ann2;
	using namespace std;

	for_each(clusters.begin(),
			 clusters.end(),
			 [&o, &normalize/*, &num*/] (ann2::cluster_samples_gen_t<Samples> &gen)
			 { gen.dump_samples(o, normalize); });

	return o;
  }

  size_t size() const {
	return std::accumulate(clusters.begin(), clusters.end(), size_t(0), 
						   [] (size_t &acc, const ann2::cluster_samples_gen_t<Samples> &g) {return acc + g.size(); } );
  }

  vector_real_t *operator[](int i) {
	using namespace boost;

	if (clusters.empty())
	  return 0;

	uniform_int<> dist(0, clusters.size() - 1 );
	variate_generator<mt19937&, uniform_int<> > die(random_gen, dist);

	int v =	die();
	dbg("Selected cluster #" << v << ", requested point #" << i << ", total clusters #" << clusters.size() );

	//	return clusters[die()][i];
	return clusters[v][i % clusters[v].size() ];
  }


  //
  // Note: can be veeery slow and memory consuming!
  //
  void shuffle() {
	using namespace std;

	for_each(clusters.begin(), clusters.end(), 
			 [] (ann2::cluster_samples_gen_t<Samples> &g) { g.shuffle(); }  );

	random_shuffle( clusters.begin(), clusters.end() );
  }
 
private:
  void init_random() {
	random_gen.seed(time(0));
  }
};


#endif
