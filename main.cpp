#include <array>
#include <iostream>
#include <sstream>
#include <string>

#include "include/utf.h"

using namespace std;

int main() {
  utf::backend b2("versions/1.12.0/lib/libtensorflow.so");
  std::cout << "TF(dll) version: " << b2.TF_Version() << std::endl;
  utf::backend b3("versions/1.14.0/lib/libtensorflow.so");
  std::cout << "TF(dll) version: " << b3.TF_Version() << std::endl;

  utf::backend::instance = &b2;


  // utf::backend b1("libtensorflow.so");

  // auto g = backend.TF_NewGraph();

  // return 0;

  // std::cout << "TF version: " << TF_Version() << std::endl;

  utf::buffer buf = read_file("models/graph.pb");

  utf::graph graph = utf::graph::from_protobuf(buf);
  utf::status status;
  utf::session_options sess_opts;
  utf::session sess{graph, sess_opts, status};
  status.check("New session");

  sess.run_targets({graph.get_op_checked("init")}, status);
  status.check("init");

  utf::initialize_variables("models/weights.json", graph, sess, status);









  //  vector<TF_Output> inputs = {graph.op_output("x")};
  //  vector<TF_Output> outputs = {graph.op_output("add")};
  //                               graph.op_output("w/Initializer/zeros")};

  vector<int64_t> dims{2};
  vector<float> values{3.0, 5.0};

  vector<utf::tensor> input_tensors;
  input_tensors.push_back(utf::tensor::create<float>(dims, values));

  vector<utf::tensor> output_tensors;

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
  cout << "aa " << output_tensors.size() << endl;

  status.check("Session RUN - w 2");

  w = output_tensors[0].get_data<float>();
  cout << "w = " << w[0] << "," << w[1] << endl;

  auto z = output_tensors[1].get_data<float>();
  cout << "z = " << z[0] << "," << z[1] << endl;

  sess.run_targets({graph.get_op_checked("train")}, status);
  output_tensors = sess.run({graph.op_output("w")}, status);
  w = output_tensors[0].get_data<float>();
  cout << "w = " << w[0] << "," << w[1] << endl;

  sess.run_targets({graph.get_op_checked("train")}, status);
  status.check("train");
  output_tensors = sess.run({graph.op_output("w")}, status);
  status.check("w");
  cout << "aa " << output_tensors.size() << endl;
  w = output_tensors[0].get_data<float>();
  cout << "w = " << w[0] << "," << w[1] << endl;

  for (int i = 0; i < 2; i++) {
    sess.run_targets({graph.get_op_checked("train")}, status);
    status.check("train");

    // output_tensors = sess.run({graph.op_output("w"), graph.op_output("beta1_power"),
    //     graph.op_output("w/Adam"), graph.op_output("w/Adam_1")}, status);
    output_tensors = sess.run({graph.op_output("w")}, status);
    status.check("w");

    cout << output_tensors.size() << "\t";
    for (auto&& x : output_tensors) {
      w = x.get_data<float>();
      cout << " " << w[0] << "," << w[1] << "\t";
    }
    cout << endl;

    // w = utf::get_data<float>(output_tensors[0]);
    // cout << "w = " << w[0] << "," << w[1] << endl;
  }

  return 0;
}
