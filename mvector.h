#ifndef MVECTOR_H // the 'include guard'
#define MVECTOR_H // see C++ Primer Sec. 2.9.2

#include <vector>

// Class that represents a mathematical vector
class MVector
{
public:
	// constructors
	MVector() {}
	explicit MVector(int n) : v(n) {}
	MVector(int n, double x) : v(n, x) {}
	MVector(std::initializer_list<double> l) : v(l) {}

	// access element (lvalue) (see example sheet 5, q5.6)
	double &operator[](int index) 
	{ 
		return v[index];
	}

	// access element (rvalue) (see example sheet 5, q5.7)
	double operator[](int index) const {
		return v[index]; 
	}

	bool operator==(const MVector &r) //comparison operator
	{
		for (unsigned u = 0; u<v.size(); u++)
		{
			if (v[u] != r[u]) {return false;} //false if any element doesn't match
		}
		return true;
	}

	int size() const { return v.size(); } // number of elements

private:
	std::vector<double> v;
};

#endif
