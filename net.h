#include <vector>
using namespace std;

#ifndef NET_H
#define NET_H

class Net
{
public:
  Net(vector<int> shape);
  Net(const vector<int> shape, const vector<double> weights);

  const vector<double>& output(const vector<double> &x);
  const vector<double>& getWeights();
  const vector<int>& getShape();
  void setBatchSize(int b);
  void HMC(double eps, int M, int L, const vector<double> &t, const vector < vector<double> > &x);

private:
  vector<double> weights;
  vector<int> shape;
  double weight_scale;
  int count;
  int batchSize;

  //allocation of memory ahead of time
  vector<double> next;
  vector<double> y;

vector<double> partialDer(const vector<double> &t, const vector  <vector<double> > &x);
  double potential(const vector<double> &t, const vector< vector<double> > &x);
  void setMaxOut();
};

#endif
