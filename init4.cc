#include <fstream>
#include "ann2.h"

#define EUCLID_CENTROID_PFX "cluster_euclid_"
#define NORM_CENTROID_PFX   "cluster_norm_"

#define START_LABEL "Start"
#define END_LABEL   "End"

#define NUMBER_OF_ITERATIONS 10000

int
main(int argc, char *argv[])
{
  using namespace std;
  using namespace ann2;
  using namespace boost;

  say("Creating sample clusters...");
  sample_clusters_db_t<10> db{
	//                  |    x1     x2   |    y1     y2      |
	cluster_bounds_vec_t{ {  0.2 ,  0.3  }, {  0.55,  0.58 } },
	cluster_bounds_vec_t{ {  0.6 ,  0.9  }, {  0.45,  0.58 } },
    cluster_bounds_vec_t{ { -0.6 , -0.9  }, { -0.45,  0.3  } },
 	cluster_bounds_vec_t{ { -1.9 , -1.2  }, { -1.3,  -1.2  } },
  };

  {
	say("Dumping the sample clusters to file...");
	ofstream fdata1("data_norm.dat");
	db.dump_clusters(fdata1, true);
	fdata1.close();

	ofstream fdata2("data.dat");
	db.dump_clusters(fdata2, false);
	fdata2.close();
  }

  boost::shared_ptr<centroid_dump_t> move_tracer(new centroid_dump_t(NUMBER_OF_ITERATIONS));

  {
	say("Creating (dot product) net...");
	cluster_net_t<Dotproduct> net_norm(2, 3);
  
	// Dump starting points
	for(size_t i = 0; i < net_norm.get_clusters_num(); i++)
	  move_tracer->dump_cluster_pos( NORM_CENTROID_PFX, i, net_norm.get_cluster_centre(i), START_LABEL );

	say("Studying...");
	net_norm.run(db, NUMBER_OF_ITERATIONS );

	// Dump resulting points
	for(size_t i = 0; i < net_norm.get_clusters_num(); i++)
	  move_tracer->dump_cluster_pos( NORM_CENTROID_PFX, size_t(i), net_norm.get_cluster_centre(i), END_LABEL );
  }


  {
	say("Creating (euclidian norm) net...");
	cluster_net_t<Euclidian> net_euclid(3, cluster_bounds_vec_t{ {-2.0, 2.0 }, { -2.0, 2.0 } } );

	for(size_t i = 0; i < net_euclid.get_clusters_num(); i++)
	  move_tracer->dump_cluster_pos( EUCLID_CENTROID_PFX, size_t(i), net_euclid.get_cluster_centre(i), START_LABEL );

	net_euclid.run(db, NUMBER_OF_ITERATIONS);

	for(size_t i = 0; i < net_euclid.get_clusters_num(); i++)
	  move_tracer->dump_cluster_pos( EUCLID_CENTROID_PFX, size_t(i), net_euclid.get_cluster_centre(i), END_LABEL );
  }

  say("Done!");
  return 0;
}
