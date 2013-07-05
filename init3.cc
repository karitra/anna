#include <cstdlib>

#include <algorithm>
#include <bitset>
#include <iterator>
#include <stdexcept>
#include <sstream> 


#include "ann2.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include <boost/numeric/ublas/vector_proxy.hpp>

#define HOR_PIXELS 7
#define VER_PIXELS 9

#define INPUT_PIXELS (HOR_PIXELS * VER_PIXELS)
#define OUTPUT_UNITS 9


#define SET_CH_BIT   'x'
#define UNSET_CH_BIT '.'

#define NET_MAX_VAL 1.0
#define NET_MIN_VAL 0.0


#define DEFAULT_EPOCHS_TO_LEARN 1000
#define DEFAULT_SAMPLES_DIR "samples1"


template<int Xin, int Yin> class digit_t;

template<int Xin1, int Yin1>
std::ostream &operator<<(std::ostream &os, digit_t<Xin1, Yin1> &d);

template<int Xin, int Yin>
class digit_t {
  int digit;
  typedef std::bitset<Xin * Yin> bitset_t;
  typedef boost::shared_ptr<bitset_t> bitset_ptr_t;
  bitset_ptr_t signal;
public:
  template<int Xin1, int Yin1>
  friend 
  std::ostream &operator<<(std::ostream &os, digit_t<Xin1, Yin1> &d);

  //	friend template<class T> void std::swap<digit_t>(digit_t &a, digit_t &b);

  digit_t( const char *fname) : 
	digit(-1),
	signal(new bitset_t)
  {
	load(fname);
  }

   
  int get_digit() const {
	return digit;
  }

  bool get_bit(int n) const {
	return signal->operator[](n);
  }

  ann2::vector_real_t get_as_vector() const {
	using namespace ann2;
	ann2::vector_real_t out( signal->size() + 1 );

	out(0) = 1.0;
	std::for_each(iter_cnt_t(0),
				  iter_cnt_t(signal->size()),
				  [&out,&signal] (int i) 
				  { 
					out( i + 1) = signal->operator[](i) ? NET_MAX_VAL : NET_MIN_VAL;
				  } );

	return out;
  }

private:

  void load(const char *fname) {
	using namespace std;

	dbg("sample file : " << fname);
	ifstream ifs(fname);
	if (!ifs)
	  throw std::runtime_error(string("Failed to open samples file ") + fname);
	
	string s;
	int row = 0;

	ifs >> digit;

	while(ifs >> s)
	  add_points(row++, s);

	ifs.close(); 				// should be on win32
  }

  void add_points(int r, const std::string &s) {
	int offset = 0;
	for_each(s.begin(), s.end(), 
			 [&r, &offset, &signal] (char ch) -> void {
			   if (r * Xin + offset > Xin * Yin) {
				 throw std::runtime_error("Signals array overflow!");
			   }
				 				 
			   if (ch == SET_CH_BIT) {
				 signal->operator[](r * Xin + offset) = true;
			   }

			   offset++;
			 } );
  }

};


template<int Xin1, int Yin1>
std::ostream &operator<<(std::ostream &os, digit_t<Xin1, Yin1> &d)
{
  os << "Digit is " << d.digit << std::endl;
  for(int j = 0; j < Yin1; j++) {
	for(int i = 0; i < Xin1; i++) {
	  os << (d.signal->operator[](j * Xin1 + i) ? SET_CH_BIT : UNSET_CH_BIT) ;
	}
	os << std::endl;
  }

  return os;
}



template<int Xin, int Yin, int Out>
class digits_generator_t : public ann2::base_sample_gen_t
{
private:  
  typedef std::vector< digit_t<Xin,Yin> > digits_vector_t;
  digits_vector_t digits;
public:

  template<int Xin1, int Yin1, int Out1>
  friend
  std::ostream &operator<<(std::ostream &o, digits_generator_t<Xin1, Yin1, Out1> &dg);


  digits_generator_t(const char *dirname) {

	for(int i = 0; i < Out; i++)
	  digits.push_back(digit_t<Xin,Yin>(get_fpath(dirname, i).c_str()));
  }

  static std::string get_fpath(const char *dirname, int i) {
	std::ostringstream os;
	os << dirname << '/' << i << ".txt";	
	return os.str();
  }
  
  size_t size() const {
	return static_cast<size_t>(Out);
  }

  void shuffle() {
	std::random_shuffle(digits.begin(), digits.end());
  }

  ann2::vector_real_t  input(size_t i) const {
	dbg("input: " << digits[i].get_digit() << " => " << digits[i].get_as_vector() );
	return digits[i].get_as_vector();
  }

  ann2::vector_real_t output(size_t i ) const {
	ann2::vector_real_t out(size() - 1 + 1);

	out(0) = 1.0; // bias (not used in output, but need it for common algorithm)

	for(unsigned int j = 1; j < size(); j++)
	  out(j) = ((unsigned int) digits[i].get_digit() == j) ? NET_MAX_VAL : NET_MIN_VAL;

	dbg("output: " << out );
	return out;
  }

};


template<int Xin, int Yin, int Out>
std::ostream &operator<<(std::ostream &o, digits_generator_t<Xin, Yin, Out> &dg)
{
  //  std::copy(dg.digits.begin(), dg.digits.end(), std::ostream_iterator< digit_t<Xin,Yin> >(o));

  std::for_each(dg.digits.begin(), dg.digits.end(),
				[&o] (digit_t<Xin, Yin> &d) -> void 
				{
				  o << d << std::endl;
				} );
  return o;
}


namespace std {
  template<int In, int Yin>
  void swap(digit_t<In,Yin> &a, digit_t<In,Yin> &b)
  {
	dbg("Digit SWAP!");
	a.signal.swap(b.signal);
	swap(a.digit, b.digit);
  }
}

int
main(int argc, char *argv[])
{
  using namespace std;
  using namespace ann2;
  namespace po = boost::program_options;
  typedef digits_generator_t<7, 9, OUTPUT_UNITS + 1> digit_gen63_9_t;

  string net_fname, samples_dir, pattern;
  int epochs, units;

  po::options_description desc("Network options");
  desc.add_options()
	("network,n", po::value<string>(&net_fname), "ANN2 Network file name" )
	//	("create,c", "Create new network ('units' must be set)" )
	("store,s",                                                
	 "\tShould we store network weights\n"
	 "Note: network file name must be set!" )
	("epochs,e", po::value<int>(&epochs), "Number of epochs to study for"                    )
	("units,u", po::value<int>(&units),   "How much units should be in (first) hidden layer" )
	("samples", po::value<string>(&samples_dir)->default_value(DEFAULT_SAMPLES_DIR), "Directory with study samples files" )
	("pattern,p", po::value<string>(&pattern), "Pattern to recognze");

  po::variables_map vm;
  try {
	po::store( po::parse_command_line(argc, argv, desc), vm );
	po::notify( vm );
  } catch(po::error &e) {
	err(e.what());
	err(desc);
	return EXIT_FAILURE;
  }
  
  network_t net;
  if (vm.count("network") && !vm.count("units")) {

	try {
	  load_net(vm["network"].as<string>().c_str(), net );
	  say("Network file with " << net.getTotalEpochs() << " study epochs was loaded!");
	} catch(boost::archive::archive_exception &ae) {
	  err("Network file " << vm["network"].as<string>().c_str() << " doesn't exist and -u option wasn't set.");
	  err(desc);
	  return EXIT_FAILURE;
	}

  } else if(vm.count("units")) {  // should create network
	units = vm["units"].as<int>();
	say("Creating net with " << units << " hidden units...");
	//net.add_layer(INPUT_PIXELS, units)->add_layer(units, OUTPUT_UNITS);
	int second_layer = 20;
	net.add_layer(INPUT_PIXELS, units)->add_layer(units, second_layer)->add_layer(second_layer,OUTPUT_UNITS);
  } else {
	err("Number of units for new network or network file to load should be set on command line.");
	err(desc);
	return EXIT_FAILURE;
  }

  if (vm.count("epochs")) { // study patterns
	say("Studying...");
	epochs = vm["epochs"].as<int>();
   
	digit_gen63_9_t dgen(samples_dir.c_str());
	//	cerr << dgen;
	feed_back_t fb(epochs);
	net.study(dgen, epochs, &fb );
	say("Number of epoch taken so far " << net.getTotalEpochs());
  }

  if (vm.count("store")) {
	if (vm.count("network")) {
	  store_net(net_fname.c_str(), net);
	} else {
	  err("Warning: network file not set, but 'store' option is set!");
	  err("         can't save network");
	}
  }


  if (vm.count("pattern")) {
	say("Recognition...");
	digit_t<7,9> d( pattern.c_str() );

	vector_real_t out = net.proc( d.get_as_vector() );
	ublas::vector_range<vector_real_t> vr(out, ublas::range(1, out.size()) );
	say( "Output for digit " << d.get_digit() << " is " << fixed << vr);
	vector_real_t::iterator item = max_element( out.begin()+1, out.end() );

	if (*item >= 0.9) {
	  say( "Max element in vector is " << *item << " with number " << (item - out.begin()) );
	} else {
	  say( "None of the units have activation more then 0.9, possible 0 (nil) value");
	}

  }

  return EXIT_SUCCESS;
}
 
