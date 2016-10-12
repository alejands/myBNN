#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "net.h"
using namespace std;

int main()
{
  //PARAMETERS FOR TRAINING ----------------------------------------------------
  //Network Structure
  int nLayers = 3;          //number of hidden layers + 2
  int shape[] = {2,10,1};   //network shape, must be size nLayers
  double weight_scale = 0.1;//range of original sampling of weights
  int batchSize = 23;       //subset picked out to estimate partial derivatives

  //HMC
  double epsillon = 0.001;   //size of microstep
  int M = 20;               //number of macrosteps
  int L = 50;               //number of microsteps (leap-frog steps)
  int numnet = 200;         //number of networks (HMC iterations)

  //Files and Data
  string netFile = "testnet"; //filename to write network to
  string trainData = "train2.dat";//training data filename
  int numTData = 10000;           //number of training data


  //CREATE AND TRAIN NETWORK ---------------------------------------------------
  cout << "Creating Network" << endl;
  vector<int> s;
  for (int i = 0; i < nLayers; i++)
    s.push_back(shape[i]);

  Net net(s);
  net.setBatchSize(batchSize);

  ifstream fin;
  ofstream fout;
  vector < vector <double> > inputs;
  vector < double > targets;
  double temp;

  //get training data
  fin.open(trainData);
  if (!fin.is_open())
  {
    cout << "Error: Training data file not found. Exiting program." << endl;
    return 0;
  }
  for (int i = 0; i < numTData; i++)
  {
    //get inputs
    vector < double > tempInp;
    for (int j = 0; j < s[0]; j++)
    {
      fin >> temp;
      tempInp.push_back(temp);
    }
    //get target
    fin >> temp;

    inputs.push_back(tempInp);
    targets.push_back(temp);
  }
  fin.close();

  //train networks
  cout << "Training Network..." << endl;
  vector < vector < double > > networks;//list of network weights from training

  for (int i = 0; i < numnet; i++)
  {
    net.HMC(epsillon,M,L,targets,inputs);
    networks.push_back(net.getWeights());

    //print progress
    cout << "Networks trained: " << i+1 << " / " << numnet << endl;
  }
  cout << "Training Complete\n" << endl;


  //WRITE NETWORK WRAPPER ------------------------------------------------------
  cout << "Writing trained network file..." << endl;
  fout.open(netFile + ".cpp");

  if(!fout.is_open())
    cout << "Error: Unable to create trained network file" << endl;

  string str;
  //header
  str = "#include <iostream>\n"
  "#include <vector>\n"
  "#include <cmath>\n"
  "using namespace std;\n\n";
  fout << str;

  //printing out trained networks
  fout << "const double w[" << numnet << "][" << networks[0].size() << "] = {";
  for (int i = 0; i < numnet; i++)
  {
    fout << "{";
    for (int j = 0; j < networks[i].size(); j++)
    {
      fout << networks[i][j];
      if (j < networks[i].size() - 1)
        fout << ", ";
    }
    fout << "}";
    if (i < numnet - 1)
      fout << ",\n";
  }
  fout << "};\n\n";

  //class declaration
  str = "class " + netFile + "\n"
  "{\n"
  "public:\n"
  "  "+netFile+"();\n"
  "  double operator()(const vector<double> &x);\n"
  "  void useLast(int n);\n\n"
  "private:\n"
  "  vector <int> shape;\n"
  "  vector < vector<double> > weights;\n"
  "  int numnet;\n"
  "  int netUsed;\n"
  "  double netOut(const vector<double> &x, int wSet);\n\n"
  "  //initilize temp vectors to avoid constant memory allocation\n"
  "  vector <double> y, next, Wi;\n"
  "};\n\n"
  "//FUCNTION DEFINITIONS "
  "---------------------------------------------------------\n"
  "//Network constructor using pretrained weights\n"
  +netFile+"::"+netFile+"()\n"
  "//Network constructor using pretrained weights\n"
  "{\n";
  fout << str;

  for (int i = 0; i < nLayers; i++)
    fout << "  shape.push_back("+to_string(s[i])+");\n";

  fout << "  numnet = "+to_string(numnet)+";\n";
  str = "  netUsed = numnet;\n\n"
  "  //placing trained weights in a vector\n"
  "  for (int i = 0; i < netUsed; i++)\n"
  "    weights.push_back(vector<double>(w[i], w[i] + sizeof w[i] / "
  "sizeof w[i][0]));\n"
  "}\n\n";
  fout << str;

  str = "double "+netFile+"::netOut(const vector<double> &x, int wSet)\n"
  "{\n"
  "  int shift = 0;\n"
  "  Wi = weights[wSet];\n\n"
  "  //copy initial input\n"
  "  y.resize(0);\n"
  "  for (int i=0; i < shape[0]; i++)\n"
  "    y.push_back(x[i]);\n\n"
  "  //calulate new layer outputs\n"
  "  for (int k=0; k < shape.size() - 1; k++)\n"
  "  {\n\n"
  "    //individual neuron values\n"
  "    next.resize(0);\n"
  "    for (int i = 0; i < shape[k+1]; i++)\n"
  "    {\n"
  "      double next_i = (Wi[shift + (shape[k]+1)*i]);\n"
  "      for (int j = 0; j < shape[k]; j++)\n"
  "        next_i += Wi[shift + (shape[k]+1)*i + j+1] * y[j];\n\n"
  "      //take tanh for all neuron except final output neurons\n"
  "      if (k < shape.size() - 2)\n"
  "        next_i = tanh(next_i);\n\n"
  "      next.push_back(next_i);\n"
  "    }\n\n"
  "    //shift over to next layer's weights\n"
  "    shift += shape[k+1] + shape[k]*shape[k+1];\n"
  "    //replace inputs to layer with outputs from layer\n"
  "    y = next;\n"
  "  }\n\n"
  "  return y[0];\n"
  "}\n\n";
  fout << str;

  str = "double "+netFile+"::operator()(const vector<double> &x)\n"
  "{\n"
  "  double output = 0.0;\n"
  "  for (int i = numnet - netUsed; i < numnet; i++)\n"
  "    output += netOut(x, i);\n"
  "  return output / netUsed;\n"
  "}\n\n";
  fout << str;

  str = "void "+netFile+"::useLast(int n)\n"
  "//use last n networks in operator()\n"
  "{\n"
  "  if (n < 1)\n"
  "    netUsed = 1;\n"
  "  else if (n > numnet)\n"
  "    netUsed = numnet;\n"
  "  else\n"
  "    netUsed = n;\n"
  "}\n\n";
  fout << str;

  cout << "Writing complete! Trained network in file:" << endl;
  cout << netFile + ".cpp" << endl;

  return 0;
}
