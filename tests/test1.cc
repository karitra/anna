#include <iostream>
#include <algorithm>

using namespace std;

template<class T, const T a = double(10.0)>
class A {
  const T d;
public:
  A() {
	cerr << "A ctor: " << a << endl;
	d = a;
  }

};

int inc(const int &i) {
  return i + 1;
}

int
main(int argc, char *argv[])
{
  using namespace std;
  int (j);

  int a[10] = {1,2,3,4,5,6,7,8,9,0};
  transform(&a[0], &a[10], &a[0],inc);

  for(int i = 0; i < 10; i++ )
	cerr << a[i] << endl;

  //  const double dd = 3.14;
  //A<double, 10.0> pi;
  return 0;
}
 
