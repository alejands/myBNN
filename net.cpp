#include <ctime>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include "net.h"
using namespace std;


Net::Net(vector<int> s) : shape(s), weights(vector<double>()), next(vector<double>()), y(vector<double>())
//constructor
//initilizes weights of network to random values between -1.0 and 1.0
{
  weight_scale = 1.0;
  batchSize = 1;
  count = 0;
  int w_size = 0;
  srand((unsigned)time(0));
  rand();

  //determine number of weights needed
  for (int i=0; i < shape.size()-1; i++)
    w_size += shape[i+1] + shape[i]*shape[i+1];

  weights.reserve(w_size);

  for (int i=0; i < w_size; i++)
    weights.push_back(2.0*weight_scale*rand()/RAND_MAX - weight_scale);

  setMaxOut();
}


Net::Net(const vector<int> s, const vector<double> w) : shape(s), weights(w),next(vector<double>()), y(vector<double>())
//constructor that takes in pre-existing weights
{
  setMaxOut();
}


void Net::setMaxOut()
{
  //determine max output size and allocate memory
  int maxOut = shape[0];
  for (int i = 1; i < shape.size(); i++)
    if (maxOut < shape[i])
      maxOut = shape[i];

  for (int i = 0; i < maxOut; i++)
  {
      next.reserve(maxOut);
      y.reserve(maxOut);
  }
}


const vector<double>& Net::output(const vector<double> &x)
{
  int shift = 0;

  //copy initial input
  y.resize(0);
  for (int i=0; i < shape[0]; i++)
    y.push_back(x[i]);

  //calulate new layer outputs
  for (int k=0; k < shape.size() - 1; k++)
  {

    //individual neuron values
    next.resize(0);
    for (int i = 0; i < shape[k+1]; i++)
    {
      double next_i = (weights[shift + (shape[k]+1)*i]);
      for (int j = 0; j < shape[k]; j++)
        next_i += weights[shift + (shape[k]+1)*i + j+1] * y[j];

      //take tanh for all neuron except final output neurons
      if (k < shape.size() - 2)
        next_i = tanh(next_i);

      next.push_back(next_i);
    }
    /*
    for (int i=0; i < shape[k+1]; i++)
    {
      next[i] = weights[shift + i];
      for (int j=0; j < shape[k]; j++)
        next[i] += weights[shift + shape[k+1]*(j+1) + i] * y[j];
    }
    */

    //shift over to next layer's weights
    shift += shape[k+1] + shape[k]*shape[k+1];
    //replace inputs to layer with outputs from layer
    y = next;
  }

  return y;
}


void Net::HMC(double eps, int M, int L, const vector<double> &t, const vector< vector<double> > &x)
//hybrid monte carlo method used for training network
{
  if (shape[shape.size() - 1] != 1)
  {
    std::cout << "Only valid for one output networks." << std::endl;
    return;
  }

  vector<double> p_i, p_f, partial;
  p_i.assign(weights.size(),0.0);
  p_f.assign(weights.size(),0.0);

  vector<double> weight_in = weights;

  //macrostep
  for (int m=0; m < M; m++)
  {
    //set all initial momenta
    for (int i = 0; i < weights.size(); i++)
      p_i[i] = -1.0 + 2.0*rand()/RAND_MAX;

    p_f = p_i;

    //microstep
    for (int l = 0; l < L; l++)
    {
      //one leap frog accross every weight
      partial = partialDer(t,x);
      for (int i = 0; i < weights.size(); i++)
      {
        p_f[i] -= (eps/2.0) * partial[i];
        weights[i] += eps * p_f[i];
      }
      partial = partialDer(t,x);
      for (int i = 0; i < weights.size(); i++)
        p_f[i] -= (eps/2.0) * partial[i];
    }//end microstep

    //get momentum mag squared
    double p_i2 = 0.0, p_f2 = 0.0;
    for (int i = 0; i < weights.size(); i++)
    {
      p_i2 += pow(p_i[i],2);
      p_f2 += pow(p_f[i],2);
    }

    //accept or reject macrostep

    double alpha = exp(p_i2/(2.0)
                    + Net(shape,weight_in).potential(t,x)
                    - p_f2/(2.0)
                    - potential(t,x));
    if ((double)rand()/RAND_MAX <= alpha)
      weight_in = weights;  //accept
    else
      weights = weight_in;  //reject

  }//end macrostep
}

/*
void Net::HMC(double eps, int M, int L, const vector<double> &t, const vector< vector<double> > &x)
{
  if (shape[shape.size() - 1] != 1)
  {
    std::cout << "Only valid for one output networks." << std::endl;
    return;
  }

  vector<double> weight_in = weights;
  int count = 0;
  //bool test = true;

  for (int m=0; m < M; m++)

    for (int i=0; i < weights.size(); i++)
    {
      double p_i = 0.2*rand()/RAND_MAX - 0.1;
      double p_f = p_i;

      for (int l=0; l < L; l++)
      {
        p_f -= (eps/2.0) * partialDer(i,count,t,x);
        count = (count + 1)%t.size();
        weights[i] += eps * p_f;
        p_f -= (eps/2.0) * partialDer(i,count,t,x);
        count = (count + 1)%t.size();
      }


      double alpha = exp(pow(p_i,2)/2.0 +
        Net(shape,weight_in).potential(t,x) - pow(p_f,2)/2.0 -
        potential(t,x));

      if (rand()/RAND_MAX <= alpha)
        weight_in = weights;
      else
        weights = weight_in;
    }
}*/

vector<double> Net::partialDer(const vector<double> &t, const vector < vector<double> > &x)
//partial derivative vector that includes partials for each weight
{
  double h = 1e-6;

  vector<double> W_plus, W_minus;
  double vPlus, vMinus, outPlus, outMinus;

  vector<double> partial;
  partial.reserve(weights.size());

  for (int k = 0; k < weights.size(); k++)
  {
    W_plus = W_minus = weights;
    W_plus[k] += h/2.0;
    W_minus[k] -= h/2.0;

    vPlus = vMinus = 0.0;

    for (int i = 0; i < batchSize; i++)
    {
      outPlus = Net(shape, W_plus).output(x[count])[0];
      outMinus = Net(shape, W_minus).output(x[count])[0];

      vPlus += pow(t[count] - outPlus,2);
      vMinus += pow(t[count] - outMinus,2);

      count = (count + 1)%t.size();
    }

    double part_i = (vPlus - vMinus)/(h)/(2 * weight_scale*weight_scale);
    //part_i += 2*weights[k]/pow(weight_scale,2);

    partial.push_back(part_i);
  }

  return partial;

  /*
  W_plus = W_minus = weights;
  W_plus[i] += h/2.0;
  W_minus[i] -= h/2.0;



  for (int i = 0; i < batchSize; i++)
  {
    double outPlus = Net(shape, W_plus).output(x[count])[0];
    double outMinus = Net(shape, W_minus).output(x[count])[0];

    vPlus += pow(t[count] - outPlus,2);
    vMinus += pow(t[count] - outMinus,2);

    count = (count + 1)%t.size();
  }

  return (vPlus - vMinus)/(h)/(2 * weight_scale*weight_scale);
  */

}


double Net::potential(const vector<double> &t, const vector< vector<double> > &x)
//potential function trying to be minimized for the network
{
  double V = 0.0;

  for (int i = 0; i < t.size(); i++)
  {
      double netOutput = output(x[i])[0];
      V += pow(t[i] - netOutput, 2);
  }

  //for (int i = 0; i < weights.size(); i++)
  //  V += pow(weights[i],2)/pow(weight_scale,2);

  return V / (2 * weight_scale*weight_scale);
}


const vector<double>& Net::getWeights()
//vector of weights
{
  return weights;
}


const vector<int>& Net::getShape()
//return shape
{
  return shape;
}

void Net::setBatchSize(int b)
{
  batchSize = b;
}
