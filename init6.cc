#include <iostream>
#include <fstream>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#define png_infopp_NULL (png_infopp) (NULL)
#define int_p_NULL (int *) (NULL)
//#include <boost/mpl/vector.hpp>
//#include <boost/gil/extension/dynamic_image/dynamic_image_all.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/gil_all.hpp>


#include "ann2.h"

#define NUMBER_OF_ITERATIONS 10000
#define NUMBER_OF_CLUSTERS   5

#define APPROX_CENTOID_PFX  "cluster_img_"

#define START_LABEL "Start"
#define END_LABEL   "End"

#define DEFAULT_IMAGE "hello.png"

using namespace boost;
using namespace boost::gil;

class image_cluster_t : public ann2::base_cluster_gen_t 
{
  //typedef boost::mpl::vector<gray8_image_t, gray16_image_t, rgb8_image_t, rgb16_image_t> img_var_t;
  //any_image<img_var_t> runtime_image;
  rgb8_image_t img;
public:

  image_cluster_t(const std::string &fname) {
	//	png_read_image(fname.c_str(), runtime_image);
	//	png_read_image(fname.c_str(), runtime_image);
	png_read_image(fname.c_str(), img);
	gen_samples(view(img));
  }

private:


  void gen_samples(const rgb8_view_t &v) {
	using namespace ann2;
	using namespace std;

	for(int y = 0; y < v.height() ; y++) {
	  for(int x = 0; x < v.width(); x++) {

		if (at_c<0>(v(x,y)) < 0x10 && 
			at_c<1>(v(x,y)) < 0x10 &&
			at_c<2>(v(x,y)) < 0x10 ) {

		  ann2::vector_real_shptr_t pv(new ann2::vector_real_t(2) );
	  
		  (*pv)(0) = (real_type) x / 10.0;
		  (*pv)(1) = (real_type) (v.height() - y) / 10.0;
		  samples.push_back( pv );
		}
		
	  }
	}
	  
  }

};

int
main(int argc, char *argv[])
{
  using namespace std;
  using namespace boost;
  using namespace ann2;
  namespace po = boost::program_options;

  int iterations, clusters;
  bool is_conv;
  std::string img_name;

  po::options_description desc("Network options");
  desc.add_options()
	("iters,i", po::value<int>(&iterations)->default_value(NUMBER_OF_ITERATIONS), "Number of iteraions to run...")
	("conv", po::value<bool>(&is_conv)->default_value(true), 
	 "Should net be used for clustering?\n"
	 "(if option not set use net for finding min/max),\n"
	 "default value is 'false' ")
	("number,n", po::value<int>(&clusters)->default_value(NUMBER_OF_CLUSTERS), "Number of clusters" )
	("file,f", po::value<std::string>(&img_name)->default_value(DEFAULT_IMAGE), "Image file to use");


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
	cluster_net_t<Euclidian> net( clusters, cluster_bounds_vec_t{ {-10.0, 10.0 }, { -10.0, 10.0 } }, is_conv );

	image_cluster_t ig(img_name);
	ofstream fdata("data_img.dat");
	ig.dump_samples(fdata);
	fdata.close();

	boost::shared_ptr<centroid_dump_t> move_tracer( new centroid_dump_t((iterations + 2) * net.get_clusters_num(), APPROX_CENTOID_PFX) );
	say("Number of study iterations: " << iterations );

	for(size_t i = 0; i < net.get_clusters_num(); i++)
	  move_tracer->dump_cluster_pos(i, net.get_cluster_centre(i), START_LABEL );

	net.run(ig, iterations, move_tracer.get() );

	for(size_t i = 0; i < net.get_clusters_num(); i++)
	  move_tracer->dump_cluster_pos(i, net.get_cluster_centre(i), END_LABEL );
  }

  return EXIT_SUCCESS;
}
 
