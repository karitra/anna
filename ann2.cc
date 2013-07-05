#include <iomanip>

#include "ann2.h"

using namespace ann2;
using namespace ublas;

unsigned long ann2::rnd_real_gen_t::rand_seed = 1024;


std::ostream &ann2::operator << (std::ostream &o, const ann2::layer_t &l)
{
  return o << *l.W;
}


void layer_t::updateWeights(const vector_real_t &d, const vector_real_t &y, const double &alpha, const double &mu) {
  using namespace std;

  matrix_real_t &Dref = *D;

  matrix_row<matrix_real_t> mr(Dref, 0);
  
  // dbg("'updateWeights' for outer layer!");
  // dbg("d => " << d);
  // dbg("y => " << y);


  // Init D first row
  for_each(iter_cnt_t(0),
		   iter_cnt_t(mr.size()),
		   [&mr,&d,&y] (int j) -> void
		   { mr(j) = (d(j) - y(j)) * y(j) * (1.0 - y(j));}  );

  // for(int j = 0; j < mr.size(); j++) {
  // 	mr(j) = (d(j) - y(j)) * y(j) * (1.0 - y(j));
  // 	// dbg(" dr(" << j << ") = " <<  d(j) - y(j) << " dy = " << y(j) * (1.0 - y(j)) << " y = " << y  ); 
  // }

  // Scale to all rows value from first one
  for_each(iter_cnt_t(1), 
		   iter_cnt_t(Dref.size1()), 
		   [&Dref, &mr] (int i) -> void
		   {
			 matrix_row<matrix_real_t> ir(Dref, i);
			 ir = mr; 
		   } );

  // for(int i = 1; i < Dref.size1(); i++ ) {
  // 	matrix_row<matrix_real_t> ir(Dref, i);
  // 	ir = mr;
  // }

  // dbg("D: " << Dref);
  // dbg("In: " << *In.get() );


  // Compute & store delta
  //matrix_real_t      &Wref  =      *W.get();
  diag_matrix_real_t &Aref  =     *In;
  matrix_real_t      &dWref = *dWprev; 

  // Prepare Ain diognal matrix
  // Note: done while studying in current version!
  // setInMatrix(Ain);

  // Count momentum and update dWprev
  dWref *= mu;
  noalias(dWref) += alpha * prod(Aref, Dref);

  dbg("Delta: " << dWref);

  // Update Weights
  dbg("Prev w: " << *W);
  W->plus_assign( dWref );
  dbg("New  w: " << *W);
}

// Update weights for hidden layer
void layer_t::updateWeights( const double &alpha, const double &mu) {
  using namespace std;

  //  dbg("'updateWeights' for hidden layer!");

  assert(next_layer);

  // Init D first row
  matrix_real_t &Dref = *D;

  matrix_row<matrix_real_t> dr(Dref, 0);
  vector_real_t &y = *next_layer->getAin();
  dbg("this layer output (input for next): " << y);

  for_each(iter_cnt_t(0),
		   iter_cnt_t(dr.size()),
		   [&next_layer,&dr,&y] (int j) -> void
		   { 
			 matrix_row<matrix_real_t> 
			   Wnext( *next_layer->getW(), j ),
			   Dnext( *next_layer->getD(), j );
			 
			 dr(j) = y(j) * (1.0 - y(j)) * ublas::inner_prod( Wnext, Dnext );
		   });

  // for( int j = 0; j < dr.size(); j++) {
  // 	matrix_row<matrix_real_t> Wnext( *next_layer->getW(), j );
  // 	matrix_row<matrix_real_t> Dnext( *next_layer->getD(), j );
  // 	dr(j) = y(j) * (1.0 - y(j)) * ublas::inner_prod( Wnext, Dnext );
  // 	//	dbg(" dr(" << j << "): prod " <<  ublas::inner_prod( Wnext, Dnext) << " dy = " << y(j) * (1.0 - y(j)) << " y = " << y(j)  );
  // }

  // Scale to all rows value from first one
  for_each(iter_cnt_t(1),
		   iter_cnt_t(Dref.size1()),
		   [&dr, &Dref] (int i) -> void
		   {
			 matrix_row<matrix_real_t> ir(Dref, i);
			 ir = dr;
		   } );

  // for(int i = 1; i < Dref.size1(); i++ ) {
  // 	matrix_row<matrix_real_t> ir(Dref, i);
  // 	ir = dr;
  // }

  // dbg("D (hidden): " << Dref);
  // dbg("In (hidden): " << *In.get() );


  // Prepare Ain diognal matrix
  // Done while studying in current version
  // setInMatrix(Ain);

  // Count momentum and update dWprev
  matrix_real_t      &dWref = *dWprev; 

  dWref *= mu;
  noalias(dWref) += alpha * prod(*In, Dref);

  dbg("Delta  (hidden): " << dWref);
  // Update Weights
  dbg("Prev w (hidden): " << *W );
  W->plus_assign( dWref );
  dbg(" New w (hidden): " << *W );
}

network_t *network_t::add_layer(int in_units, int out_units, bool biased, activation_base_real_t *af) throw(std::invalid_argument) 
{
  if (biased) in_units++, out_units++;

  if (layers.size() &&
	  layers[ layers.size() - 1 ]->getOutUnitsNum() != in_units ) // check last layer for compatibility with new
	throw std::invalid_argument("Input/output units number mismatch!");


  boost::shared_ptr<activation_base_real_t> afp(af ? af : new sigmoid_real_fa_t);

  layer_smart_ptr_t lp( new layer_t(in_units, out_units, 0, afp) );
  layers.push_back( lp );

  // set 'next' pointer for previous layer
  if (layers.size() >= 2)
	layers[layers.size() - 2]->setNext( lp.get() );

  return this;  
} 


real_type network_t::sample_error(const vector_real_t &d, const vector_real_t &y) {
	vector_real_t diff = d - y;
	return ublas::inner_prod(diff, diff) / 2.0;
}


int network_t::study(base_sample_gen_t &sgen,
					 const int epoch_limit, 
					 net_mentor_base_t *vizier,
					 const real_type &epoch_err_limit,
					 const real_type &net_err_limit)
{
  unsigned int epoch = 0;
  real_type epoch_err;
  vector_real_t y, d;

  init_study_vars();

  do {
	
	epoch_err = 0.0;

	// for each sample in sample set
	for(size_t i = 0; i < sgen.size(); i++ ) {

	  y = proc_study( sgen.input(i) );
	  d = sgen.output(i);

	  epoch_err += sample_error(d,y);

	  dbg("study   y: " << y      );
	  dbg("desired d: " << d      );


	  // output layer
	  layers.back()->updateWeights(d, y);

	  // hidden layers
	  for_each(layers.rbegin() + 1, layers.rend(), 
			 [] (layer_smart_ptr_t &lr) -> void { lr->updateWeights(); } );

	} // for each sample in sample set

	sgen.shuffle();
	if (vizier)
	  vizier->updateFeedBack();

	life_cycle_epochs++;

  } while(not stop_condition(d, y,
							 epoch, epoch_limit, epoch_err, epoch_err_limit, 
							 net_err_limit) );

  delete_study_vars();
  
  return epoch;
}

bool network_t::stop_condition(const vector_real_t &d,
							   const vector_real_t &y,
							   unsigned int &epoch,
							   const unsigned int epoch_limit,
							   const real_type &epoch_err,
							   const real_type &epoch_err_limit,
							   const real_type &net_err_limit) 
{
  // First of all: epoch finished!
  epoch++;
  
  dbg("epoch is #" << epoch);
  
  if (epoch >= epoch_limit)
	return true;

  double gross_error;

  total_sqr_err += epoch_err;
  gross_error    = total_sqr_err / epoch;

  dbg("gross err: " << gross_error                      );
  dbg("prev  err: " << prev_epoch_err                   );
  dbg("epoch err: " << epoch_err                        );
  dbg("err  diff: " << fabs(epoch_err - prev_epoch_err) );
  dbg("err bound: " << epoch_err_limit                  );

  //if (gross_error < net_err_limit)
  //	return true;

  if (fabs(epoch_err - prev_epoch_err) < epoch_err_limit &&
   	  epoch_err < net_err_limit) {
	dbg("Covergence at epoch #" << epoch << '!');
   	return true; // Covergence!
  }

  prev_epoch_err = epoch_err;
  return false;
}


void
ann2::store_net(const char *fname, const ann2::network_t &net) 
{
  std::ofstream ofs2(fname);
  boost::archive::text_oarchive oa(ofs2);

  dbg("Store net");
  oa << net;
  dbg("Serialization done!");

  ofs2.close(); //avoiding win32 bug!
}


void 
ann2::load_net(const char *fname, ann2::network_t &net) 
{
  std::ifstream ifs2(fname);
  boost::archive::text_iarchive ia(ifs2);

  dbg("Loading net");
  ia >> net;
  dbg("Loading done!");

  ifs2.close(); //avoiding win32 bug!
}


namespace ann2 {


  template<>
  void
  cluster_net_t<Euclidian>::initW(const cluster_bounds_vec_t *const bounds)
  {
	using namespace boost;
	using namespace std;

	boost::mt19937 rnd_gen( time(0) );

	for_each(iter_cnt_t( 0 ), iter_cnt_t( W->size2() ),

			 [&W,&rnd_gen,&bounds] (int j) -> void {
			   matrix_column<matrix_real_t> c( *W, j);

			   //	   generate( iter_cnt_t(0), iter_cnt_t( W->size1() ),
			   for_each( iter_cnt_t(0), iter_cnt_t( W->size1() ),
						 [&rnd_gen, &bounds, &c] (int i) -> void {
						   using namespace boost;
						   real_type 
							 from = (bounds) ? (*bounds)[i].first  : 0.0,
							 to   = (bounds) ? (*bounds)[i].second : 0.1;

						   if (from >= to)
							 throw runtime_error("min >= max in distribution bounds!");

						   uniform_real<> dist( from, to );
						   variate_generator<boost::mt19937&, uniform_real<> > gen(rnd_gen, dist);
						   
						   c(i) = gen();
						   //						   return gen();
						 } );

			 } );
  }
  
  template<>
  void
  cluster_net_t<Dotproduct>::initW(const cluster_bounds_vec_t *const /* unused */)
  {
	using namespace ublas;
	using namespace std;
	
	rnd_real_gen_t rnd( -1.0, 1.0 );

	for_each( iter_cnt_t(0), 
			  iter_cnt_t(W->size2()),
			  [&W, &rnd] (int j) -> void {
				
				matrix_column<matrix_real_t> w(*W, j);			  
				generate(w.begin(), w.end(), rnd );
				
				// normolize
				noalias(w) = w * (1/norm_2(w));
			  } ) ;
  }

  template<>
  size_t
  cluster_net_t<Euclidian>::proc(const vector_real_t &x) 
  {
	using namespace std;
	using namespace ublas;

	std::vector<real_type> euclid_norms;

	// find winner: min distance
	for_each( iter_cnt_t( 0 ),
			  iter_cnt_t( W->size2() ),
			  [&x, &euclid_norms, this] (int j) -> void {
				matrix_column<matrix_real_t> c( *W, j);
				euclid_norms.push_back( norm_2(c - x) );
			  } );
	
	// Update winner
	return update_winner(x,  (min_element(euclid_norms.begin(), 
										  euclid_norms.end  () ) - euclid_norms.begin() ), false);
  }

  template<>
  size_t
  cluster_net_t<Dotproduct>::proc(const vector_real_t &x1) 
  {
	using namespace ublas;
	// Note: W should be normolized already!
	vector_real_t x = x1 * (1/norm_2(x1));

	dbg("Getting y");
	*y = prod( x, *W );
	return update_winner(x,
						 std::max_element( y->begin(), y->end() ) - y->begin(),
						 true);
  }

}

