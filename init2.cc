#include <cmath>
#include <cstdlib>

#include <iostream>
#include <fstream>

#include <boost/progress.hpp>
#include <boost/timer.hpp>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "ann2.h"

#define STEPS_NUMBER 100
#define DUMMY_NETWORK_FILE_NAME "--"

#define UNITS_NUMBER       10
#define STUDY_ITERATIONS 1000

const double PI = 4.0 * atan(1.0);

void 
approx(const int inner_num,
	   const unsigned int test_points,
	   ann2::network_t &net, 
	   ann2::approx_rect &arec,
	   const int layers_num=2)
	   // const ann2::real_type &x1,
	   // const ann2::real_type &x2,
	   // const ann2::real_type &y1,
	   // const ann2::real_type &y2 ) 
{
  using namespace std;
  using namespace ann2;

  ostringstream os;
  os << "approx" << inner_num << '_' << test_points << ".dat";
  ofstream of( os.str().c_str() );
  if (!of)
	throw std::runtime_error("Failed to open approximation file!");
  
  vector_real_t             in(2),     out(2);
  scale_to_one_t<real_type> sx(arec.x1, arec.x2), sy(arec.y1, arec.y2);
  scale_back_t<real_type>   mx(arec.x1, arec.x2), my(arec.y1, arec.y2);

  double step = (arec.x2 - arec.x1) / STEPS_NUMBER;
  feed_back_t progress2( STEPS_NUMBER );
  boost::timer t;
  for(double dx = arec.x1; dx < arec.x2; dx += step) {
	
	in(0) = 1.0;
	in(1) = sx(dx);
	out   = net.proc(in, &progress2);

	of << dx << ' ' << sin(dx) << ' ' << my(out(1)) << endl;
  }
  say("\n" "Recognition process takes " << t.elapsed() << " seconds.");
  of.close();
}

int
main(int argc, char *argv[])
{
  using namespace std;
  using namespace ann2;
  namespace po = boost::program_options;

  std::string net_file, store_file;
  po::options_description op("Options");

  int 
	inner_layer_units       = 0,
	study_iterations        = 0;
  const int samples_number  = 18;

  bool 
	//should_learn = false, 
	net_created  = false;
	
  op.add_options() 
	("network,n", po::value<string>(&net_file)->default_value(DUMMY_NETWORK_FILE_NAME), "Network file to load"               )
	("units,u",   po::value<int>(&inner_layer_units)->default_value(UNITS_NUMBER),      "Number of units in hidden layer"    )
	("epoch,e",   po::value<int>(&study_iterations)->default_value(STUDY_ITERATIONS),   "Epoch limit"                        )
	("store,s",   po::value<string>(&store_file),                                       "Storing network to file"            )
	("learn,l",   
	 "Should learning process be started\n"
	 "Only valid if net is loaded from file" )
	("help,h",   "Print usage and version" );

  po::variables_map vm;
  try {
	po::store( po::parse_command_line(argc, argv, op), vm );
	po::notify( vm );
  } catch(po::unknown_option &o) {
	cerr << o.what() << endl; 
	cerr << op       << endl;
	return EXIT_FAILURE;
  }

  if (vm.count("help")) {
	cout << op << endl;
	return EXIT_SUCCESS;
  }

  approx_rect arec(PI, 3 * PI, -1.0, 1.0);

  sinus_generator_t<double, samples_number> gen(arec);
  gen.dump_original_samples(); // For debug and compare!

  say("Creating network...");
  shared_ptr<network_t> net( new network_t );
  if (vm.count("network") && vm["network"].as<string>() != string(DUMMY_NETWORK_FILE_NAME) ) {
	std::string fname = vm["network"].as<string>();
	try { 
	  say("Loading network from file " <<  vm["network"].as<string>() );
	  load_net(fname.c_str(), *net.get() );
	} catch(...) { // TODO: catch appropriate exception!
	  err("Failed to load network file!");
	  throw;
	}

  } else {
	net_created = true;
   	net->add_layer(1, inner_layer_units)->add_layer(inner_layer_units, 1);
  }


  if (net_created || vm.count("learn")) { 
	//
	// Values for 10 samples and 10000 iterations:
	//
	// 3  - curve doesn't have 'covergence' on the right bound;
	// 6  - the same as for 3;
	// 10 -
	// 12 - the same as for 6, but line is much better on the right bound, but not a sinus yet;
	// 24 - much better, but still not ideal;
	// Values for 12 samples and 10000 iterations:
	// 9  - well, some sisible difference at maximums;
	// 30 - no, not an ideal;
	// Values for 12 samples and 100000 iterations:
	// 100 - sigmoid?;
	// Values for 32 samples and 10000 iterations:
	// 60 - some mismatch at peaks;
	// Values for 24 samples and 1000 iterations:
	// 3  - bad result - sigmoid
	say("Studying for "   << net->get_demension(0).second <<
		" hidden units, " << samples_number               <<
		" samples and "   << study_iterations             << " iterations ..." );

	int n = 0;
	boost::timer t;
	{
	  feed_back_t progress1(study_iterations);
	  t.restart();
	  n = net->study( gen, study_iterations, &progress1 );
	}

	say(endl << "Study process takes " << t.elapsed() << " seconds and " << n << " epoch" << (n ? "s" : "")  );
  }

  if ( vm.count("store") ) {
	//	store_net( vm["store"].as<string>().c_str(), *net.get() );
	store_net( store_file.c_str(), *net.get() );
  }

  say( "Total epochs taken so far " << net->getTotalEpochs());

  say("Approximation...");
  approx(inner_layer_units, gen.size(), *net.get(), arec );

  return EXIT_SUCCESS;
}
