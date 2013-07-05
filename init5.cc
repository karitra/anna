#include <fstream>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include "ann2.h"

#define NUMBER_OF_ITERATIONS 10000
#define NUMBER_OF_CLUSTERS   3

#define APPROX_CENTOID_PFX  "cluster_par_"

#define START_LABEL "Start"
#define END_LABEL   "End"


int
main(int argc, char *argv[])
{
  using namespace std;
  using namespace boost;
  using namespace ann2;
  namespace po = boost::program_options;

  int iterations, clusters;
  bool is_conv;

  po::options_description desc("Network options");
  desc.add_options()
	("iters,i", po::value<int>(&iterations)->default_value(NUMBER_OF_ITERATIONS), "Number of iteraions to run...")
	("conv", po::value<bool>(&is_conv)->default_value(true), 
	 "Should net be used for clustering?\n"
	 "(if option not set use net for finding min/max),\n"
	 "default value is 'false' ")
	("number,n", po::value<int>(&clusters)->default_value(NUMBER_OF_CLUSTERS), "Number of clusters" );

  po::variables_map vm;
  try {
	po::store( po::parse_command_line(argc, argv, desc), vm );
	po::notify( vm );
  } catch(po::unknown_option &o) {
	err(o.what());
	err(desc);
	return EXIT_FAILURE;
  }
  
  {
	say("Creating 'extremums' net with " << clusters <<" (is_conv=" << is_conv << ")...");
	cluster_net_t<Euclidian> net( clusters, cluster_bounds_vec_t{ {-2.0, 2.0 }, { -2.0, 2.0 } }, is_conv );

	parabolic_generator_t<64> pg{ 
	  cluster_bounds_vec_t{ 
		// x1   x2							
		{ 0.4, 2.0 },
		// a   b  in y = (x + a)^2 + b;
		{ -1.0, 0.8 }
	  } };

	ofstream fdata("data_par.dat");
	pg.dump_samples(fdata);
	fdata.close();


	boost::shared_ptr<centroid_dump_t> move_tracer( new centroid_dump_t((iterations + 2) * net.get_clusters_num(), APPROX_CENTOID_PFX) );
	say("Number of study iterations: " << iterations );

	for(size_t i = 0; i < net.get_clusters_num(); i++)
	  move_tracer->dump_cluster_pos(i, net.get_cluster_centre(i), START_LABEL );

	net.run(pg, iterations, move_tracer.get() );

	for(size_t i = 0; i < net.get_clusters_num(); i++)
	  move_tracer->dump_cluster_pos(i, net.get_cluster_centre(i), END_LABEL );
  }

  return EXIT_SUCCESS;
}
 
