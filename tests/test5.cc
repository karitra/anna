#include <iostream>

enum Those { None, Foo, Bar};


class A {
public:

  template<Those T=None>
  void foo(int j) {
	//	T i;
	std::cerr << "<>foo\n";
  }

};

template<>
void A::foo<Foo>(int j) {
  std::cerr  << "Foo\n";
}


template<>
void A::foo<Bar>(int j) {
  std::cerr  << "Bar\n";
}

template<Those T=None>
class B {
public:
  void bar(int j);
};

template<>
void 
B<Foo>::bar(int j) 
{
  std::cerr << "B<Foo>::bar\n";
}

template<>
void 
B<None>::bar(int j) 
{
  std::cerr << "B<Foo>::bar\n";
}


int
main(int argc, char *argv[])
{
  A a;
  a.foo<Foo>(10);
  a.foo<Bar>(11);
  a.foo<>(12);

  B<> b;
  b.bar(32);
  return 0;
}
 
