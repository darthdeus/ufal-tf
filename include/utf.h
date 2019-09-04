#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include <tensorflow/c/c_api.h>

#include "utils.h"

namespace utf {
using namespace std;

DEF_HANDLE(buffer, Buffer)
DEF_HANDLE(session_options, SessionOptions)
DEF_HANDLE(import_graph_def_options, ImportGraphDefOptions)

template <typename T>
T* get_data(TF_Tensor* tensor) {
  return reinterpret_cast<T*>(TF_TensorData(tensor));
}

class status : public handle<TF_Status, TF_DeleteStatus> {
 public:
  status() : handle(TF_NewStatus()) {}

  void check(const std::string& str) {
    TF_Code code = TF_GetCode(obj);

    if (code == 0) {
      std::cout << str << " OK" << std::endl;
    } else {
      const char* message = TF_Message(obj);

      std::stringstream ss;

      ss << "ERR " << str << std::endl;
      ss << "Code: " << code << std::endl << message << std::endl;

      std::cerr << ss.str();
      throw std::runtime_error(ss.str());
    }
  }
};

class graph;

class operation {
 public:
  TF_Operation* op;
  explicit operation(TF_Operation* op) : op(op) {}

  TF_Output output(int index = 0) {
    TF_Output output;
    output.oper = op;
    output.index = index;

    return output;
  }
};

class graph : public handle<TF_Graph, TF_DeleteGraph> {
 public:
  graph() : handle(TF_NewGraph()) {}

  operation get_op_checked(const std::string& name) {
    operation op{TF_GraphOperationByName(obj, name.c_str())};

    if (!op.op) {
      std::stringstream ss;
      ss << "Failed to find op '" << name << "' in the graph." << std::endl;

      std::cerr << ss.str();
      throw std::runtime_error(ss.str());
    }

    return op;
  }

  TF_Output op_output(const std::string& name, int output = 0) {
    // TODO: tady se to smaze moc brzo
    return get_op_checked(name).output(output);
  }

  static graph from_protobuf(buffer& buf) {
    graph g;
    status s;
    import_graph_def_options opts;

    TF_GraphImportGraphDef(g.obj, buf.obj, opts.obj, s.obj);
    s.check("Graph import");

    return g;
  }

  TF_Output get_initializer(TF_Operation* op) {
    string name = TF_OperationName(op);
    string target = name + "/Initializer";

    size_t pos = 0;
    TF_Operation* oper;

    while ((oper = TF_GraphNextOperation(obj, &pos)) != nullptr) {
      string other_name = TF_OperationName(oper);

      if (other_name.substr(0, target.size()) == target) {
        cout << "Found initializer " << name << " -> " << other_name << endl;
        return operation{oper}.output(0);
      }
    }

    throw std::runtime_error("Initializer missing for variable" + name);
  }
};

void TF_DeleteSessionNoStatus(TF_Session* sess) {
  status s;
  TF_DeleteSession(sess, s.obj);

  // TODO: check status
}

class tensor;

template <typename T>
TF_Tensor* allocate_uninitialized_tensor(vector<int64_t> dims) {
  int64_t items = 1;
  for (auto d : dims) {
    items *= d;
  }

  // TODO: TF_NewTensor vs TF_AllocateTensor

  //    TF_AllocateTensor(TF_FLOAT, dims, 1, dims[0] *
  //    TF_DataTypeSize(TF_FLOAT)), TF_AllocateTensor(TF_FLOAT, dims, 1,
  //    dims[0] * TF_DataTypeSize(TF_FLOAT))

  TF_DataType dtype = type_to_tf_datatype<T>();
  int64_t bytes = items * TF_DataTypeSize(dtype);

  TF_Tensor* tensor = TF_AllocateTensor(dtype, dims.data(), dims.size(), bytes);

  return tensor;
}

template <typename T>
TF_Tensor* allocate_tensor(vector<int64_t> dims, vector<T> values) {
  // TODO: check and delete
  //    TF_AllocateTensor(TF_FLOAT, dims, 1, dims[0] *
  //    TF_DataTypeSize(TF_FLOAT))

  TF_Tensor* tensor = allocate_uninitialized_tensor<T>(dims);

  auto tensor_data = reinterpret_cast<T*>(TF_TensorData(tensor));
  std::copy(values.begin(), values.end(), tensor_data);

  return tensor;
}

class tensor : public handle<TF_Tensor, TF_DeleteTensor> {
 public:
  explicit tensor(TF_Tensor* t) : handle(t) {}

  template <typename T>
  T* get_data() {
    return reinterpret_cast<T*>(TF_TensorData(obj));
  }

  template <typename T>
  static tensor create_uninitialized(vector<int64_t> dims) {
    TF_Tensor* t = allocate_uninitialized_tensor<T>(dims);

    return tensor{t};
  }

  template <typename T>
  static tensor create(vector<int64_t> dims, vector<T> values) {
    TF_Tensor* t = allocate_tensor(dims, values);

    return tensor{t};
  }
};

struct session : public handle<TF_Session, TF_DeleteSessionNoStatus> {
  graph& g;

  session(graph& g, session_options& sopts, status& s)
      : g(g), handle(TF_NewSession(g.obj, sopts.obj, s.obj)) {}

  vector<TF_Tensor*> map_tensors(vector<tensor>& tensors) {
    vector<TF_Tensor*> results;
    results.reserve(tensors.size());

    for (tensor& t : tensors) {
      results.push_back(t.obj);
    }

    return results;
  }

  void run_targets(initializer_list<operation> target_ops, status& status) {
    vector<TF_Operation*> ops;
    ops.reserve(target_ops.size());
    for (auto& op : target_ops) {
      ops.push_back(op.op);
    }

    TF_SessionRun(obj, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0,
                  ops.data(), target_ops.size(), nullptr, status.obj);
  }

  vector<TF_Tensor*> run(initializer_list<TF_Output> outputs, status& status) {
    vector<TF_Output> no_inputs{};
    vector<utf::tensor> no_input_tensors{};
    vector<TF_Operation*> no_targets;
    vector<TF_Output> outputs_{outputs.begin(), outputs.end()};

    return run(no_inputs, no_input_tensors, outputs_, no_targets, status);
  }
  //
  //  vector<TF_Tensor*> run(initializer_list<TF_Output> inputs,
  //                         vector<tensor>& input_values,
  //                         initializer_list<TF_Output> outputs, status&
  //                         status) {
  //    return run({inputs.begin(), inputs.end()}, input_values,
  //               {outputs.begin(), outputs.end()}, {}, status);
  //  }

  void run(vector<TF_Output>& inputs, vector<tensor>& input_values,
           vector<TF_Operation*>& target_ops, status& status) {
    vector<TF_Output> no_outputs;

    run(inputs, input_values, no_outputs, target_ops, status);
  }

  vector<TF_Tensor*> run(vector<TF_Output>& inputs,
                         vector<tensor>& input_values,
                         vector<TF_Output>& outputs,
                         vector<TF_Operation*> target_ops, status& status) {
    TF_Operation* const target_opers[] = {};

    vector<TF_Tensor*> output_values;
    output_values.reserve(outputs.size());

    TF_SessionRun(obj, nullptr, inputs.data(), map_tensors(input_values).data(),
                  inputs.size(), outputs.data(), output_values.data(),
                  outputs.size(), target_ops.data(), target_ops.size(), nullptr,
                  status.obj);

    return output_values;
  }
};

//  tensor()
//      : dims(dims),
//        len(len),
//        t(allocate_uninitialized_tensor<T>(&*dims.begin(), len)){};
//
//  tensor(std::initializer_list<int64_t> dims, int64_t len,
//         std::initializer_list<T> values)
//      : dims(dims),
//        len(len),
//        t(allocate_tensor<T>(&*dims.begin(), len, values)){};

// template <typename T>
// class typed_tensor {
// public:

//
//  typed_tensor(const typed_tensor& rhs) = delete;
//  typed_tensor& operator=(const typed_tensor& rhs) = delete;
//
//  typed_tensor(typed_tensor&& rhs) noexcept
//      : t(rhs.t), dims(rhs.dims), len(rhs.len) {
//    rhs.t = nullptr;
//  }
//  typed_tensor& operator=(typed_tensor&& rhs) noexcept {
//    t = rhs.t;
//    dims = rhs.dims;
//    len = rhs.len;
//    rhs.t = nullptr;
//  }
//
//  ~tensor() {
//    if (t) {
//      TF_DeleteTensor(t);
//    }
//  }
//};

}  // namespace utf

void heap_buf_deallocator(void* data, size_t length) { free(data); }

utf::buffer read_file(const char* fname) {
  FILE* f = fopen(fname, "rb");

  fseek(f, 0, SEEK_END);
  auto size = ftell(f);
  fseek(f, 0, SEEK_SET);

  void* data = malloc(size);
  fread(data, 1, size, f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = size;
  buf->data_deallocator = heap_buf_deallocator;

  return utf::buffer{buf};
}
