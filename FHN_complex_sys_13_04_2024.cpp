#include "mvector.h"
#include "mmatrix.h"

#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <string>
#include <vector>
#include <stdexcept>
#include <typeinfo>


// globals
double AMP = 5, ALPHA = 1, BETA = 1, GAMMA = 1, STEP = 5e-5, TIME = 1e-2, MANUAL_DIFF_STEP = 1e-4, DIFF_COEFF = 1;
int SEGMENTS = 10;
double PULSE_TIME=2;

////////////////////////////////////////////////////////////////////////////////
// Set up random number generation

// Set up a "random device" that generates a new random number each time the program is run
std::random_device rand_dev;

// Set up a pseudo-random number generater "rnd", seeded with a random number
//std::mt19937 rnd(rand_dev());
std::mt19937 rnd(1);

// MMatrix * MVector
MVector operator*(const MMatrix &m, const MVector &v)
{
	assert(m.Cols() == v.size());

	MVector r(m.Rows());

	for (int i=0; i<m.Rows(); i++)
	{
		for (int j=0; j<m.Cols(); j++)
		{
			r[i]+=m(i,j)*v[j];
		}
	}
	return r;
}

// transpose(MMatrix) * MVector
MVector TransposeTimes(const MMatrix &m, const MVector &v)
{
	assert(m.Rows() == v.size());

	MVector r(m.Cols());

	for (int i=0; i<m.Cols(); i++)
	{
		for (int j=0; j<m.Rows(); j++)
		{
			r[i]+=m(j,i)*v[j];
		}
	}
	return r;
}

// MVector + MVector
MVector operator+(const MVector &lhs, const MVector &rhs)
{
	assert(lhs.size() == rhs.size() && "V+V");

	MVector r(lhs.size(), 0);
	for (int i=0; i<lhs.size(); i++)
		r[i] = lhs[i] + rhs[i];

	return r;
}

// MVector - MVector
MVector operator-(const MVector &lhs, const MVector &rhs)
{
	assert(lhs.size() == rhs.size() && "V-V");

	MVector r(lhs.size(), 0);
	for (int i=0; i<lhs.size(); i++)
		r[i] = lhs[i] - rhs[i];

	return r;
}


// MMatrix = MVector <outer product> MVector
// M = a <outer product> b
MMatrix OuterProduct(const MVector &a, const MVector &b)
{
	MMatrix m(a.size(), b.size());
	for (int i=0; i<a.size(); i++)
	{
		for (int j=0; j<b.size(); j++)
		{
			m(i,j) = a[i]*b[j];
		}
	}
	return m;
}

// Hadamard product
MVector operator*(const MVector &a, const MVector &b)
{
	assert(a.size() == b.size());
	
	MVector r(a.size());
	for (int i=0; i<a.size(); i++)
		r[i]=a[i]*b[i];
	return r;
}

// double * MMatrix
MMatrix operator*(double d, const MMatrix &m)
{
	MMatrix r(m);
	for (int i=0; i<m.Rows(); i++)
		for (int j=0; j<m.Cols(); j++)
			r(i,j)*=d;

	return r;
}

// double * MVector
MVector operator*(double d, const MVector &v)
{
	MVector r(v);
	for (int i=0; i<v.size(); i++)
		r[i]*=d;

	return r;
}

// Mvector + const
MVector operator+(MVector &v, const double &d)
{
    MVector out(v.size());
    for (int i = 0; i < v.size(); i++)
    {
        out[i] = v[i] + d;
    }
    return out;
}

// MVector -= MVector
MVector operator-=(MVector &v1, const MVector &v)
{
	assert(v1.size()==v.size());
	
	for (int i=0; i<v1.size(); i++)
		v1[i]-=v[i];
	
	return v1;
}

// MVector += MVector
MVector operator+=(MVector &v1, const MVector &v)
{
	assert(v1.size()==v.size() && "MVector += MVector");
	
	for (int i=0; i<v1.size(); i++)
		v1[i]+=v[i];
	
	return v1;
}

// MMatrix -= MMatrix
MMatrix operator-=(MMatrix &m1, const MMatrix &m2)
{
	assert (m1.Rows() == m2.Rows() && m1.Cols() == m2.Cols() && "M-=M");

	for (int i=0; i<m1.Rows(); i++)
		for (int j=0; j<m1.Cols(); j++)
			m1(i,j)-=m2(i,j);

	return m1;
}

// MMatrix += MMatrix
MMatrix operator+=(MMatrix &m1, const MMatrix &m2)
{
	assert (m1.Rows() == m2.Rows() && m1.Cols() == m2.Cols());

	for (int i=0; i<m1.Rows(); i++)
		for (int j=0; j<m1.Cols(); j++)
			m1(i,j)+=m2(i,j);

	return m1;
}


// Output function for MVector
inline std::ostream &operator<<(std::ostream &os, const MVector &rhs)
{
	std::size_t n = rhs.size();
	os << "(";
	for (std::size_t i=0; i<n; i++)
	{
		os << rhs[i];
		if (i!=(n-1)) os << ", ";
	}
	os << ")";
	return os;
}

// Output function for MMatrix
inline std::ostream &operator<<(std::ostream &os, const MMatrix &a)
{
	int c = a.Cols(), r = a.Rows();
	for (int i=0; i<r; i++)
	{
		os<<"(";
		for (int j=0; j<c; j++)
		{
			os.width(10);
			os << a(i,j);
			os << ((j==c-1)?')':',');
		}
		os << "\n";
	}
	return os;
}

void print(const MVector &a)
{
    std::cout << a << std::endl;
}

double Max(const double &x, const double &y)
{
    if (x >= y) {return x;}
    return y;
}

std::vector<MVector> FHN_sys(const MVector &v, const MVector &w, const MVector &e, const MVector &I, const MVector &a, const MVector &b, const MVector &g)
{
    if (v.size() != w.size()) 
    {   throw std::invalid_argument("mismatched v & w sizes in FHN_sys");   }
    int length = v.size();

    MVector dv(length, 0.0), dw(length, 0.0);
    for (int i = 0; i < length; i++)
    {
        dv[i] = v[i] * (a[i] - v[i]) * (v[i] - 1) - w[i] - I[i];
        dw[i] = e[i] * (b[i] * v[i] - g[i] * w[i]);
    }
    std::vector<MVector> out = {dv, dw};
    return out;
}

std::vector<MVector> FHN_base(const MVector &v_init, const MVector &epsilon, const MVector &I, const MVector &a, const MVector &b, const MVector &g, const double &step, const double &time)
{   
    int total_steps = static_cast<int>(time/step), length = v_init.size();
    MVector v = v_init, w(length, 0.0);

    for (int i =0; i < total_steps; i++)
    {
        std::vector<MVector> difs = FHN_sys(v, w, epsilon, I, a, b, g);
        v = v + (step * difs[0]);
        w = w + (step * difs[1]); 
    }
    std::vector<MVector> out = {v, w};
    return out;
}

std::vector<MVector>  FHN_find_equilibrium(const MVector &I, const MVector &a, const MVector &b, const MVector &g)
{
    MVector v={0};
    double step = 1e-2, time=1;
    MVector e = {20};

    return FHN_base(v, e, I, a, b, g, step, time);
}

MVector ddifs(const MVector x)
{
    if (x.size() < 2) {throw std::invalid_argument("ddifs too short");}

    int length = x.size();
    MVector out(length);
    out[0] = x[1] - x[0];
    out[length-1] = x[length-2]-x[length-1];

    for (int i = 1; i < length-1; i ++)
    {
        out[i] = x[i-1] + x[i+1] - 2*x[i];
    }
    return out;
}

std::vector<MVector> FHN_pulse(const MVector &v, const MVector &w, const MVector &e, const MVector &I, const MVector &a, const MVector &b, const MVector &g, const double &pulse_amp)
{
    if (v.size() != w.size()) 
    {   throw std::invalid_argument("mismatched v & w sizes in FHN_sys");   }
    int length = v.size();

    MVector dv(length, 0.0), dw(length, 0.0);

    dv[0] = v[0] * (a[0] - v[0]) * (v[0] - 1) - w[0] - (I[0] - pulse_amp);
    dw[0] = e[0] * (b[0] * v[0] - g[0] * w[0]);

    for (int i = 1; i < length; i++)
    {
        dv[i] = v[i] * (a[i] - v[i]) * (v[i] - 1) - w[i] - I[i];
        dw[i] = e[i] * (b[i] * v[i] - g[i] * w[i]);
    }
    std::vector<MVector> out = {dv, dw};
    return out;
}

std::vector<MVector> FHN_complex(const MVector &v_init, const MVector &epsilon, const MVector &I, const MVector &a, const MVector &b, const MVector &g, const double &step, const double &time)
{
    std::vector<MVector> state = FHN_find_equilibrium(I, a, b, g);
    double v_base = state[0][0], w_base = state[1][0];
    MVector v_out(v_init.size()), w_out(v_init.size());
    int total_steps = static_cast<int>(time/step);
    std::vector<MVector> difs;
    double v_last;
    v_last = v_base*1.001;

    for (int i = 0; i < v_init.size(); i++)
    {
        MVector v(SEGMENTS, v_base), w(SEGMENTS, w_base);
        MVector e_current(SEGMENTS, epsilon[i]), I_current(SEGMENTS, I[i]), a_current(SEGMENTS, a[i]), b_current(SEGMENTS, b[i]), g_current(SEGMENTS, g[i]);
        int t;
        t = static_cast<int>(PULSE_TIME / step);

        for (int s = 0; s < t; s++)
        {
            
            difs = FHN_pulse(v, w, e_current, I_current, a_current, b_current, g_current, v_init[i]);

            if (s%50==0)
            {
            std::cout << "v:  " << v << std::endl;
            //std::cout << "w:  " << w << std::endl;
            //std::cout << "difs:  " << difs[0] << std::endl;
            //std::cout << "ddifs:  " << ddifs(v) << std::endl;
            }

            v = v + step * ( difs[0] + DIFF_COEFF*ddifs(v) );
            w = w + step * difs[1];
            v_last = Max(v_last, v[SEGMENTS-1]);            
        }

        for (int s = t; s <= total_steps; s++)
        {
            
            difs = FHN_sys(v, w, e_current, I_current, a_current, b_current, g_current);

            if (s%50==0)
            {
            std::cout << s << "v:  " << v << std::endl;
            //std::cout << "w:  " << w << std::endl;
            //std::cout << "difs:  " << difs[0] << std::endl;
            //std::cout << "ddifs:  " << ddifs(v) << std::endl;
            }
            v = v + step * ( difs[0] + DIFF_COEFF*ddifs(v) );
            w = w + step * difs[1];
            v_last = Max(v_last, v[SEGMENTS-1]);            
        }

        v_out[i] = v_last;
        w_out[i] = w[SEGMENTS-1];
    }
    std::vector<MVector> out = {v_out, w_out};
    return out;
}


// Generate 1000 points of test data in a spiral pattern
void GetSpiralData(std::vector<MVector> &x, std::vector<MVector> &y, int points)
{
	std::mt19937 lr;
	x = std::vector<MVector>(points, MVector(2));
	y = std::vector<MVector>(points, MVector(1));

	double twopi = 8.0*atan(1.0);
	for (int i=0; i<points; i++)
	{
		x[i]={lr()/static_cast<double>(lr.max()),lr()/static_cast<double>(lr.max())};
		double xv=x[i][0]-0.5, yv=x[i][1]-0.5;
		double ang = atan2(yv,xv)+twopi;
		double rad = sqrt(xv*xv+yv*yv);

		double r=fmod(ang+rad*20, twopi);
		y[i][0] = (r<0.5*twopi)?1:-1;
	}
}

MVector safe_subtract(const MVector left, const MVector &right, double factor, const double minimum)
{
    MVector temp(left.size(), 0.0);
    double val;
    for (int l = 0; l < left.size(); l++)
    {
        val = left[l] - (factor * right[l]);
        if (val < 0) {val = Max(left[l]/2, minimum);}
        temp[l] = val;
    }
    return temp;
}

MMatrix safe_subtract(const MMatrix left, const MMatrix &right, double factor, const double minimum)
{
    MMatrix temp(left.Rows(), left.Cols(), 0.0);
    double val;
    for (int r = 0; r < left.Rows(); r++)
    {   
        for (int c = 0; c < left.Cols(); c++)
        {val = left(r,c) - (factor * right(r,c));
        if (val < 0) {val = Max(left(r,c)/2, minimum);}
        temp(r,c) = val;}
    }
    return temp;
}

class Network
{
public:

    int nlayers;
    std::vector<int> nneurons;
    std::vector<MVector> Is, epsilons, alphas, betas, gammas, biases, errors, Is_diff, epsilons_diff, alphas_diff, betas_diff, gammas_diff, biases_diff, FHNs, FHNs_delta_I, FHNs_delta_e, FHNs_delta_a, FHNs_delta_alpha, FHNs_delta_b, FHNs_delta_g, activations;
    MVector inputs;
    std::vector<MMatrix> weights, weights_diff;
    
    Network(std::vector<int> neurons_)
    {
        nneurons = neurons_; 
        nlayers = neurons_.size();
        Is = std::vector<MVector> {};
        epsilons = std::vector<MVector> {};
        alphas = std::vector<MVector> {};
        betas = std::vector<MVector> {};
        gammas = std::vector<MVector> {};
        weights = std::vector<MMatrix> {};
        biases = std::vector<MVector> {};
        errors = std::vector<MVector> {};
        Is_diff = std::vector<MVector> {};
        epsilons_diff = std::vector<MVector> {};
        alphas_diff = std::vector<MVector> {};
        betas_diff = std::vector<MVector> {};
        gammas_diff = std::vector<MVector> {};
        weights_diff = std::vector<MMatrix> {};
        biases_diff = std::vector<MVector> {};
        FHNs = std::vector<MVector> {};
        FHNs_delta_I = std::vector<MVector> {};
        FHNs_delta_e = std::vector<MVector> {};
        FHNs_delta_a = std::vector<MVector> {};
        FHNs_delta_alpha = std::vector<MVector> {};
        FHNs_delta_b = std::vector<MVector> {};
        FHNs_delta_g = std::vector<MVector> {};
        activations = std::vector<MVector> {};
        inputs = MVector {}; 
    }

    void init_params()
    {
        std::normal_distribution<> dist(0.5, 0.5);
        std::uniform_real_distribution<> distu(0.1,0.5);
        for (int l = 0; l < nlayers - 1; l++)
        {
            int neurons = nneurons[l], neurons_next = nneurons[l+1];
            MVector temp_e(neurons);
            MVector temp_I(neurons);
            MVector temp_a(neurons);
            MVector temp_bet(neurons);
            MVector temp_g(neurons);
            MVector temp_b(neurons_next);
            MMatrix temp_w(neurons, neurons_next);

            for (int n = 0; n < neurons; n++)
            {
                temp_e[n] = 0.01; //distu(rnd); !!constant!!
                temp_I[n] = dist(rnd);
                temp_bet[n] = dist(rnd);
                temp_g[n] = dist(rnd);
                temp_a[n] = dist(rnd);

                for (int m = 0; m < neurons_next; m++)
                {
                    temp_w(n, m) = dist(rnd);
                    temp_b[m] = dist(rnd);
                }
            }
            epsilons.push_back(temp_e);
            Is.push_back(temp_I);
            alphas.push_back(temp_a);
            betas.push_back(temp_bet);
            gammas.push_back(temp_g);
            biases.push_back(temp_b);
            weights.push_back(temp_w);
        }
    }

    MVector feed_forward(const MVector &x)
    {
        if (x.size() != nneurons[0]) 
    {   throw std::invalid_argument("mismatched x & input sizes in feed forward");   }
        inputs = x;
        activations = std::vector<MVector> {};
        activations.push_back(x);
        FHNs = std::vector<MVector> {};
        FHNs_delta_I = std::vector<MVector> {};
        FHNs_delta_e = std::vector<MVector> {};
        FHNs_delta_a = std::vector<MVector> {};
        FHNs_delta_alpha = std::vector<MVector> {};
        FHNs_delta_b = std::vector<MVector> {};
        FHNs_delta_g = std::vector<MVector> {};

        for (int l = 0; l < nlayers - 1; l++)
        {
            MVector FHN = FHN_base(activations[l], epsilons[l], Is[l], alphas[l], betas[l], gammas[l], STEP, TIME)[0];
            MVector FHN_delta_I = FHN_base(activations[l], epsilons[l], Is[l] + MANUAL_DIFF_STEP, alphas[l], betas[l], gammas[l], STEP, TIME)[0];
            MVector FHN_delta_e = FHN_base(activations[l], epsilons[l] + MANUAL_DIFF_STEP, Is[l], alphas[l], betas[l], gammas[l], STEP, TIME)[0];
            MVector FHN_delta_a = FHN_base(activations[l] + MANUAL_DIFF_STEP, epsilons[l], Is[l], alphas[l], betas[l], gammas[l], STEP, TIME)[0];
            MVector FHN_delta_alpha = FHN_base(activations[l], epsilons[l], Is[l], alphas[l] + MANUAL_DIFF_STEP, betas[l], gammas[l], STEP, TIME)[0];
            MVector FHN_delta_b = FHN_base(activations[l], epsilons[l], Is[l], alphas[l], betas[l] + MANUAL_DIFF_STEP, gammas[l], STEP, TIME)[0];
            MVector FHN_delta_g = FHN_base(activations[l], epsilons[l], Is[l], alphas[l], betas[l], gammas[l] + MANUAL_DIFF_STEP, STEP, TIME)[0];

            FHNs.push_back(FHN);
            FHNs_delta_I.push_back(FHN_delta_I);
            FHNs_delta_e.push_back(FHN_delta_e);
            FHNs_delta_a.push_back(FHN_delta_a);
            FHNs_delta_alpha.push_back(FHN_delta_alpha);
            FHNs_delta_b.push_back(FHN_delta_b);
            FHNs_delta_g.push_back(FHN_delta_g);
            activations.push_back(TransposeTimes(weights[l], FHN) + biases[l]);
        }

        return activations[activations.size() - 1];
    }

    double cost(const MVector &logits, const MVector &y)
    {
        double total = 0;
        for (int i = 0; i < logits.size(); i++)
        {
            total += (logits[i] - y[i]) * (logits[i] - y[i]);
        }
        total /= 2 * y.size();
        return total;
    }

    double cost_total(const std::vector<MVector> &arr_x, const std::vector<MVector> &arr_y)
    {
        double total = 0;
        for (int i = 0; i < arr_x.size(); i++)
        {
            total += cost(feed_forward(arr_x[i]), arr_y[i]);
        }
        return total;
    }

    MVector cost_diff(const MVector &logits, const MVector &y)
    {
        MVector temp(y.size());
        for (int i = 0; i < y.size(); i++)
        {
            temp[i] = logits[i] - y[i];
        }
        return temp;
    }

    double accuracy_1D(const std::vector<MVector> &arr_x, const std::vector<MVector> &arr_y)
    {
        assert(arr_x.size() == arr_y.size());
        double total = 0;
        int acc;

        for (int i = 0; i < arr_x.size(); i++)
        {   
            acc = static_cast<int>((feed_forward(arr_x[i])[0] - arr_y[i][0]) < 0.5);
            total += acc;
        }
        return total/arr_x.size();
    }

    void clear_diffs()
    {   
        Is_diff = std::vector<MVector> {};
        epsilons_diff = std::vector<MVector> {};
        biases_diff = std::vector<MVector> {};
        weights_diff = std::vector<MMatrix> {};
        alphas_diff = std::vector<MVector> {};
        betas_diff = std::vector<MVector> {};
        gammas_diff = std::vector<MVector> {};

        for (int l = 0; l < nlayers-1; l++)
        {
            int neurons = nneurons[l], neurons_next = nneurons[l+1];

            MVector zero_Ie(neurons, 0.0);
            MVector zero_b(neurons_next, 0.0);
            MMatrix zero_w(neurons, neurons_next, 0.0);

            Is_diff.push_back(zero_Ie);
            epsilons_diff.push_back(zero_Ie);
            alphas_diff.push_back(zero_Ie);
            betas_diff.push_back(zero_Ie);
            gammas_diff.push_back(zero_Ie);
            biases_diff.push_back(zero_b);
            weights_diff.push_back(zero_w);
        }
    }

    void test_diff(double x)
    {
        epsilons_diff[0][0] = x;
    }

    void backpropagate(const MVector &x, const MVector &y)
    {   
        feed_forward(x);
        std::vector<MVector> errors(nlayers-1);

        MVector error = cost_diff(activations[activations.size()-1], y);
        errors[errors.size()-1] = error;
        biases_diff[biases_diff.size()-1] += error;
        weights_diff[weights_diff.size()-1] += OuterProduct(activations[activations.size()-2], error);
   
        MVector WTD = weights[weights.size()-1] * error;
        MVector dfde = (1/MANUAL_DIFF_STEP) * (FHNs_delta_e[FHNs_delta_e.size()-1] - FHNs[FHNs.size()-1]);
        MVector dfdI = (1/MANUAL_DIFF_STEP) * (FHNs_delta_I[FHNs_delta_I.size()-1] - FHNs[FHNs.size()-1]);
        MVector dfdalpha = (1/MANUAL_DIFF_STEP) * (FHNs_delta_I[FHNs_delta_alpha.size()-1] - FHNs[FHNs.size()-1]);
        MVector dfdb = (1/MANUAL_DIFF_STEP) * (FHNs_delta_I[FHNs_delta_b.size()-1] - FHNs[FHNs.size()-1]);
        MVector dfdg = (1/MANUAL_DIFF_STEP) * (FHNs_delta_I[FHNs_delta_g.size()-1] - FHNs[FHNs.size()-1]);
        epsilons_diff[epsilons_diff.size()-1] += dfde * WTD; 
        Is_diff[Is_diff.size()-1] += dfdI * WTD;
        alphas_diff[Is_diff.size()-1] += dfdalpha * WTD;
        betas_diff[Is_diff.size()-1] += dfdb * WTD;
        gammas_diff[Is_diff.size()-1] += dfdg * WTD;

        int maxl = nlayers-2;
        if (maxl < 0) {maxl=0;}
        for (int l = maxl; l > 0; l--)
        {
            MVector dfda = (1/MANUAL_DIFF_STEP) * (FHNs_delta_a[l] - FHNs[l]);
            error = (weights[l] * error) * dfda;
            errors[l] = error;
            biases_diff[l-1] += error;
            weights_diff[l-1] += OuterProduct(error, activations[l-1]).T();

            WTD = weights[l-1] * error;
            dfde = (1/MANUAL_DIFF_STEP) * (FHNs_delta_e[l-1] - FHNs[l-1]);
            dfdI = (1/MANUAL_DIFF_STEP) * (FHNs_delta_I[l-1] - FHNs[l-1]);
            dfdalpha = (1/MANUAL_DIFF_STEP) * (FHNs_delta_alpha[l-1] - FHNs[l-1]);
            dfdb = (1/MANUAL_DIFF_STEP) * (FHNs_delta_b[l-1] - FHNs[l-1]);
            dfdg = (1/MANUAL_DIFF_STEP) * (FHNs_delta_g[l-1] - FHNs[l-1]);
            epsilons_diff[l-1] += dfde * WTD;
            Is_diff[l-1] += dfdI * WTD;
            alphas_diff[l-1] += dfdalpha * WTD;
            betas_diff[l-1] += dfdb * WTD;
            gammas_diff[l-1] += dfdg * WTD;
        }
    }

    void train(const std::vector<MVector> &arr_x, const std::vector<MVector> &arr_y, double lr)
    {
        int size = arr_x.size();
        clear_diffs();

        for (int i = 0; i < arr_x.size(); i++)
        {   
            backpropagate(arr_x[i], arr_y[i]);
        }

        for (int l = 0; l < nlayers-2; l++)
        {
            weights[l] -= (lr/size) * weights_diff[l];
            biases[l] -=  (lr/size) * biases_diff[l];
            //epsilons[l] -=  (lr/size) * epsilons_diff[l];
            Is[l] -=  (lr/size) * Is_diff[l];
            alphas[l] -=  (lr/size) * alphas_diff[l];
            betas[l] -=  (lr/size) * betas_diff[l];
            gammas[l] -=  (lr/size) * gammas_diff[l];

            //weights[l] = safe_subtract(weights[l], weights_diff[l], lr/size);
            //biases[l] = safe_subtract(biases[l], biases_diff[l], lr/size);
            // !!turned off!! epsilons[l] = safe_subtract(epsilons[l], epsilons_diff[l], lr/size, 1e-2);
            //Is[l] = safe_subtract(Is[l], Is[l], lr/size);
            //alphas[l] = safe_subtract(alphas[l], alphas_diff[l], lr/size, 1e-15);
            //betas[l] = safe_subtract(betas[l], betas_diff[l], lr/size, 1e-15);
            //gammas[l] = safe_subtract(gammas[l], gammas_diff[l], lr/size, 1e-15);
        }
        clear_diffs();
    }

};

int main_()
{
    Network n({2,5,5,1});
    n.init_params();
    MVector input(2, 5.0);
    std::cout << n.feed_forward(input) << std::endl;

    std::vector<MVector> X_train, Y_train;
    std::uniform_real_distribution<> dist_u(-1.0, 1.0);
    for (int i = 0; i <20; i++)
    {
        double val1 = dist_u(rnd), val2 = dist_u(rnd);
        MVector x = {val1, val2}, y = {static_cast<double>(static_cast<int>(val1*val1 + val2*val2 < 1))};
        X_train.push_back(x);
        Y_train.push_back(y);
    }

    std::cout << n.cost_total(X_train, Y_train) << std::endl;
    return 1;
}

int main__()
{
    std::vector<MVector> X_train, Y_train;
    GetSpiralData(X_train, Y_train, 100);

    std::vector<MVector> X_test, Y_test;
    GetSpiralData(X_test, Y_test, 20);

    std::vector<int> l_size = {5, 10, 15, 20, 30};

    for (int ls = 0; ls < l_size.size(); ls++)
    {
    Network net({2, l_size[ls], l_size[ls], 1});

    std::vector<double> costs, accuracies;
    double cost, acc;

    net.init_params();

    int BATCH_SIZE = 2, ITERS = 501, count;
    double LR = 5e-5;

    for (int i = 0; i <ITERS; i++)
    {
        count = 0;
        while (count < X_train.size())
        {
            std::vector<MVector>::const_iterator first_x = X_train.begin() + count;
            std::vector<MVector>::const_iterator last_x = X_train.begin() + count + BATCH_SIZE;
            std::vector<MVector>::const_iterator first_y = Y_train.begin() + count;
            std::vector<MVector>::const_iterator last_y = Y_train.begin() + count + BATCH_SIZE;
            std::vector<MVector> X_sub(first_x, last_x);
            std::vector<MVector> Y_sub(first_y, last_y);

            net.train(X_sub, Y_sub, LR);
            count += BATCH_SIZE;
        }

        if (i % 10 == 0)
        {      
            cost = net.cost_total(X_test, Y_test) ;
            acc = net.accuracy_1D(X_test, Y_test);
            costs.push_back(cost);
            accuracies.push_back(acc);
            std::cout << "Iter: " << i << ", cost: " << cost << ", acc: " << acc << std::endl;
        }
    }

    std::string base = "C:\\Users\\Andrzej Socha\\Desktop\\FHN NN Results\\Base\\spiral 500 iters lr 5e-5 ", file_name;
    file_name = std::to_string(l_size[ls]);

    std::ofstream f(base + file_name + " neurons.txt");
    f << "[" << "(" << costs[0] << ", " << accuracies[0] << ")";
    for (int i = 1; i < costs.size(); i++)
    {
        f << ", (" << costs[i] << ", " << accuracies[i] << ")";
    }
    f << "]";
    f.close();

    }

}


int main_test1()
{
    MVector e = {0.01}, a = {3}, I={-0.1}, b={2}, g={1}, v={10};
    double t = 150, step =0.02, time;

    for (int i = 1; i < 2; i++)
    {
    time = t*i;
    std::cout << FHN_complex(v, e, I, a, b, g, step, time)[0] << std::endl;
    }
    return 1;
}

int main_test2()
{
    MVector e = {0.01}, a = {2}, I={0.1}, b={2}, g={2}, v={-10}, w={0};
    std::cout << FHN_sys(v, w, e, I, a, b, g)[0];
}

int main()
{
    double pi=3.14159;
    std::vector<MVector> X_train, Y_train;
    std::uniform_real_distribution<> dist_u(-1.5, 1.5);
    for (int i = 0; i <100; i++)
    {
        double val1 = dist_u(rnd), val2 = dist_u(rnd);
        MVector x = {val1, val2}, y = {static_cast<double>(static_cast<int>(std::sin(pi * val1) * std::sin(pi * val2) < 0))};
        X_train.push_back(x);
        Y_train.push_back(y);
    }

    std::vector<MVector> X_test, Y_test;
    for (int i = 0; i <100; i++)
    {
        double val1 = dist_u(rnd), val2 = dist_u(rnd);
        MVector x = {val1, val2}, y = {static_cast<double>(static_cast<int>(std::sin(pi * val1) * std::sin(pi * val2) < 0))};
        X_test.push_back(x);
        Y_test.push_back(y);
    }

    //std::string dataset = "C:\\Users\\Andrzej Socha\\Desktop\\FHN NN Results\\Base\\square\\dataset.txt";
    //std::ofstream file(dataset);
    //for (int t = 0; t < X_train.size(); t++)
    //{
    //    file << "" << X_train[t] << ", " << Y_train[t] << "_ ";
    //}
    //file.close();


    std::vector<int> l_size = {30};

    for (int ls = 0; ls < l_size.size(); ls++)
    {
    std::cout << l_size[ls] << std::endl;
    Network net({2, l_size[ls], l_size[ls], 1});

    std::vector<double> costs, accuracies, costs_recent;
    double cost, acc;

    net.init_params();
    for (int l = 0; l < net.nlayers-1; l++)
    {
        std::cout << "wieghts " << l << ": " << net.weights[l] << "\n";
        std::cout << "biases " << l << ": " << net.biases[l] << "\n";
        std::cout << "epsilons " << l << ": " << net.epsilons[l] << "\n";
        std::cout << "Is " << l << ": " << net.Is[l] << "\n";
        std::cout << "alphas " << l << ": " << net.alphas[l] << "\n";
        std::cout << "betas " << l << ": " << net.betas[l] << "\n";
        std::cout << "gammas " << l << ": " << net.gammas[l] << "\n";
    }

    int BATCH_SIZE = 10, ITERS = 5001, count;
    double LR = 0.02, recent_size;

    for (int i = 0; i <ITERS; i++)
    {
        count = 0;
        while (count < X_train.size())
        {
            std::vector<MVector>::const_iterator first_x = X_train.begin() + count;
            std::vector<MVector>::const_iterator last_x = X_train.begin() + count + BATCH_SIZE;
            std::vector<MVector>::const_iterator first_y = Y_train.begin() + count;
            std::vector<MVector>::const_iterator last_y = Y_train.begin() + count + BATCH_SIZE;
            std::vector<MVector> X_sub(first_x, last_x);
            std::vector<MVector> Y_sub(first_y, last_y);

            net.train(X_sub, Y_sub, LR);
            count += BATCH_SIZE;
        }

        //std::ofstream x("C:\\Users\\Andrzej Socha\\Desktop\\FHN NN Results\\Base\\data2.txt", std::ios_base::app);
        //x << i << "\n";
        //for (int l = 0; l < net.nlayers-1; l++)
        //{
        //    x << "wieghts " << l << ": " << net.weights[l] << "\n";
        //    x << "biases " << l << ": " << net.biases[l] << "\n";
        //    x << "epsilons " << l << ": " << net.epsilons[l] << "\n";
        //    x << "Is " << l << ": " << net.Is[l] << "\n";
        //    x << "alphas " << l << ": " << net.alphas[l] << "\n";
        //    x << "betas " << l << ": " << net.betas[l] << "\n";
        //    x << "gammas " << l << ": " << net.gammas[l] << "\n";
        //}
        //x << "----------------------------------------------------------------" << "\n";
        //x.close();

        if (i % 10 == 0)
        {      
            cost = net.cost_total(X_test, Y_test);
            acc = net.accuracy_1D(X_test, Y_test);
            if (cost != cost)
            {
                for (int l = 0; l < net.nlayers-1; l++)
                {
                    std::cout << "wieghts " << l << ": " << net.weights[l] << "\n";
                    std::cout << "biases " << l << ": " << net.biases[l] << "\n";
                    std::cout << "epsilons " << l << ": " << net.epsilons[l] << "\n";
                    std::cout << "Is " << l << ": " << net.Is[l] << "\n";
                    std::cout << "alphas " << l << ": " << net.alphas[l] << "\n";
                    std::cout << "betas " << l << ": " << net.betas[l] << "\n";
                    std::cout << "gammas " << l << ": " << net.gammas[l] << "\n";
                }
            }

            //recent_size = costs_recent.size();
            //if (recent_size > 3)
            //{
            //    if ( cost > (costs_recent[recent_size-1]+costs_recent[recent_size-2]+costs_recent[recent_size-3]+costs_recent[recent_size-4])/4) {LR /= 2; costs_recent = {};}
            //}

            costs.push_back(cost);
            //costs_recent.push_back(cost);
            accuracies.push_back(acc);
            std::cout << "Iter: " << i << ", cost: " << cost << ", acc: " << acc << ", lr: " << LR << std::endl;
        }
    }

    std::string base = "C:\\Users\\Andrzej Socha\\Desktop\\FHN NN Results\\Base\\square\\expanded model round 1, 5000 iters, 100 test, lr 2e-3 squares ", file_name;
    file_name = std::to_string(l_size[ls]);

    std::ofstream f(base + file_name + " neurons.txt");
    f << "[" << "(" << costs[0] << ", " << accuracies[0] << ")";
    for (int i = 1; i < costs.size(); i++)
    {
        f << ", (" << costs[i] << ", " << accuracies[i] << ")";
    }
    f << "]";
    f.close();


    //std::string boundary = "C:\\Users\\Andrzej Socha\\Desktop\\FHN NN Results\\Base\\expanded model, 30 n, 100 test, equivalent of 1000 iters, boundary round.txt";
    MVector dec;
    std::ofstream fb("C:\\Users\\Andrzej Socha\\Desktop\\FHN NN Results\\Base\\square\\boundary\\expanded model," +  file_name + "n, 100 test, equivalent of 5000 iters, boundary square.txt");
    for (int x = 0; x <= 50; x++)
    {
        for (int y = 0; y <= 50; y++)
        {
            dec = {(x*0.06)-1.5, (y*0.06)-1.5};
            fb << net.feed_forward(dec) << ", "; 
        }
    }
    fb.close();
    }
}