#include <array>
#include <iostream>
#include <sstream>
#include <string>

#include "include/json.hpp"
#include "include/utf.h"

using namespace std;

int main() {
  std::cout << "TF version: " << TF_Version() << std::endl;

  utf::buffer buf = read_file("models/graph.pb");

  utf::graph graph = utf::graph::from_protobuf(buf);
  utf::status status;
  utf::session_options sess_opts;
  utf::session sess{graph, sess_opts, status};

  status.check("New session");

  vector<TF_Output> no_outputs;


  ifstream is("models/weights.json");
  using json = nlohmann::json;

  json j;
  is >> j;

  vector<TF_Output> init_inputs;
  vector<utf::tensor> init_input_tensors;
  vector<TF_Operation*> init_targets;

  for (auto& var : j["variables"]) {
    string name = var["name"];

    auto shape = var["shape"].get<vector<int64_t>>();
    auto values = var["values"].get<vector<float>>();

    init_inputs.push_back(graph.get_initializer(graph.get_op_checked(name).op));
    init_targets.push_back(graph.get_op_checked(name + "/Assign").op);

    init_input_tensors.push_back(utf::tensor::create<float>(shape, values));
    cout << "var\t" << var["name"] << "\t" << var["shape"] << " ... " << shape[0] << endl;
  }


//  size_t pos = 0;
//  TF_Operation* oper;
//
//  cout << "*************" << endl;
//  while ((oper = TF_GraphNextOperation(graph.obj, &pos)) != nullptr) {
//    cout << TF_OperationName(oper) << endl;
//  }
//  cout << "*************" << endl;
//
//  graph.get_initializer(graph.get_op_checked("w").op);
//
//      sess.run({graph.op_output("w/Initializer/ones")}, init_tensors,
//               no_outputs, {graph.get_op_checked("w/Assign").op}, status);

  sess.run(init_inputs, init_input_tensors, no_outputs, init_targets, status);

  status.check("Session RUN - init all");











  //  vector<TF_Output> inputs = {graph.op_output("x")};
  //  vector<TF_Output> outputs = {graph.op_output("add")};
  //                               graph.op_output("w/Initializer/zeros")};

  vector<int64_t> dims{2};
  vector<float> values{3.0, 5.0};

  vector<utf::tensor> input_tensors;
  input_tensors.push_back(utf::tensor::create<float>(dims, values));

  vector<TF_Tensor*> output_tensors;

  vector<utf::tensor> no_tensors;

  float* w;

  //  output_tensors = sess.run(inputs, input_tensors, outputs, status);
  //  status.check("Session RUN");
  //
  //  float* t1 = utf::get_data<float>(output_tensors[0]);
  //  float* t2 = utf::get_data<float>(output_tensors[1]);
  //
  //  cout << "add = " << t1[0] << " " << t1[1] << endl;
  //  cout << "add = " << t2[0] << " " << t2[1] << endl;

  //  sess.run_targets({graph.get_op_checked("init")}, status);
  //  status.check("Session RUN - init");
  //
  //  output_tensors = sess.run({graph.op_output("w")}, status);
  //  status.check("Session RUN - w");
  //
  //  w = utf::get_data<float>(output_tensors[0]);
  //  cout << "w = " << w[0] << "," << w[1] << endl;

  //
  //  w = utf::get_data<float>(output_tensors[0]);
  //  cout << "w = " << w[0] << "," << w[1] << endl;

  //  output_tensors = sess.run({graph.op_output("w")}, input_tensors,
  //                            {graph.op_output("w/read")}, status);

  output_tensors = sess.run({graph.op_output("w"), graph.op_output("z")}, status);

  status.check("Session RUN - w 2");

  w = utf::get_data<float>(output_tensors[0]);
  cout << "w = " << w[0] << "," << w[1] << endl;

  auto z = utf::get_data<float>(output_tensors[1]);
  cout << "z = " << z[0] << "," << z[1] << endl;

  return 0;
}
