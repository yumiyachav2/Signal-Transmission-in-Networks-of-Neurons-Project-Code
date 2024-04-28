#ifndef MMATRIX_H // the 'include guard'
#define MMATRIX_H

#include <vector>
#include <iostream>

// Class that represents a mathematical matrix
class MMatrix
{
public:
	// constructors
	MMatrix() : nRows(0), nCols(0) {}
	MMatrix(int n, int m, double x = 0) : nRows(n), nCols(m), A(n * m, x) {}

	// set all matrix entries equal to a double
	MMatrix &operator=(double x)
	{
		for (unsigned i = 0; i < nRows * nCols; i++) A[i] = x;
		return *this;
	}

	// access element, indexed by (row, column) [rvalue]
	double operator()(int i, int j) const
	{
		return A[j + i * nCols];
	}

	// access element, indexed by (row, column) [lvalue]
	double &operator()(int i, int j)
	{
		return A[j + i * nCols];
	}

	// size of matrix
	int Rows() const { return nRows; }
	int Cols() const { return nCols; }

	bool operator==(const MMatrix &r) //comparison operator
	{
		int row, col;
		for (unsigned u=0; u<A.size(); u++)
		{	
			col = u % r.Cols();
			row = (u - col) / r.Cols();
			if (A[u] != r(row ,col)) {return false;} //false if any element doesn't match
		}
		return true;
	}

	MMatrix T()
	{
		MMatrix temp(nCols, nRows, 0.0);
		for (int n = 0; n < nRows; n++)
		{
			for (int m = 0; m < nCols; m++)
			temp(m, n) = A[m + n * nCols];
		}
		return temp;
	}

private:
	unsigned int nRows, nCols;
	std::vector<double> A;
};

#endif
