#pragma once

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "tensorflow/c/c_api.h"

#include "json.hpp"
#include "utils.h"

namespace utf {

using namespace std;

template<typename T>
class TD;

#define UTF_STRINGIFY2(X) #X
#define UTF_STRINGIFY(X) UTF_STRINGIFY2(X)

#define UTF_GET_SYM(NAME) \
  { \
      this->NAME = (decltype(this->NAME)) dlsym(dl, UTF_STRINGIFY(NAME)); \
      if (!this->NAME) { \
        throw std::runtime_error(std::string{dlerror()}); \
      } \
  }


class backend {
  public:
  const char* (*TF_Version)();

  TF_Buffer* (*TF_NewBuffer)();
  void (*TF_DeleteBuffer)(TF_Buffer*);

  TF_SessionOptions* (*TF_NewSessionOptions)();
  void (*TF_DeleteSessionOptions)(TF_SessionOptions*);

  TF_ImportGraphDefOptions* (*TF_NewImportGraphDefOptions)();
  void (*TF_DeleteImportGraphDefOptions)( TF_ImportGraphDefOptions* opts);

  void* (*TF_TensorData)(const TF_Tensor*);
  const char* (*TF_Message)(const TF_Status* s);
  TF_Code (*TF_GetCode)(const TF_Status* s);

  TF_Status* (*TF_NewStatus)();
  void (*TF_DeleteStatus)(TF_Status*);

  TF_Graph* (*TF_NewGraph)();
  void (*TF_DeleteGraph)(TF_Graph*);
  TF_Operation* (*TF_GraphOperationByName)(TF_Graph* graph, const char* oper_name);
  void (*TF_GraphImportGraphDef)(TF_Graph* graph, const TF_Buffer* graph_def, const TF_ImportGraphDefOptions* options, TF_Status* status);
  const char* (*TF_OperationName)(TF_Operation* oper);
  TF_Operation* (*TF_GraphNextOperation)(TF_Graph* graph, size_t* pos);

  size_t (*TF_DataTypeSize)(TF_DataType dt);
  TF_Tensor* (*TF_AllocateTensor)(TF_DataType, const int64_t* dims, int num_dims, size_t len);
  void (*TF_DeleteTensor)(TF_Tensor*);

  TF_Session* (*TF_NewSession)(TF_Graph* graph, const TF_SessionOptions* opts, TF_Status* status);
  void (*TF_DeleteSession)(TF_Session*, TF_Status* status);
  void (*TF_SessionRun)(
    TF_Session* session,
    // RunOptions
    const TF_Buffer* run_options,
    // Input tensors
    const TF_Output* inputs, TF_Tensor* const* input_values, int ninputs,
    // Output tensors
    const TF_Output* outputs, TF_Tensor** output_values, int noutputs,
    // Target operations
    const TF_Operation* const* target_opers, int ntargets,
    // RunMetadata
    TF_Buffer* run_metadata,
    // Output status
    TF_Status*);


  backend(std::string dll_fname) {
    // TODO dlfree
    auto dl = dlopen(dll_fname.c_str(), RTLD_NOW);

    if (!dl) {
      throw std::runtime_error(std::string{dlerror()});
    }

    UTF_GET_SYM(TF_Version);

    UTF_GET_SYM(TF_NewBuffer);
    UTF_GET_SYM(TF_DeleteBuffer);

    UTF_GET_SYM(TF_NewSessionOptions);
    UTF_GET_SYM(TF_DeleteSessionOptions);

    UTF_GET_SYM(TF_NewImportGraphDefOptions);
    UTF_GET_SYM(TF_DeleteImportGraphDefOptions);

    UTF_GET_SYM(TF_TensorData);
    UTF_GET_SYM(TF_Message);
    UTF_GET_SYM(TF_GetCode);

    UTF_GET_SYM(TF_NewStatus);
    UTF_GET_SYM(TF_DeleteStatus);

    UTF_GET_SYM(TF_NewGraph);
    UTF_GET_SYM(TF_DeleteGraph);
    UTF_GET_SYM(TF_GraphOperationByName);
    UTF_GET_SYM(TF_GraphImportGraphDef);
    UTF_GET_SYM(TF_OperationName);
    UTF_GET_SYM(TF_GraphNextOperation);

    UTF_GET_SYM(TF_DataTypeSize);
    UTF_GET_SYM(TF_AllocateTensor);
    UTF_GET_SYM(TF_DeleteTensor);

    UTF_GET_SYM(TF_NewSession);
    UTF_GET_SYM(TF_DeleteSession);
    UTF_GET_SYM(TF_SessionRun);

    // this->TF_Version = (decltype(this->TF_Version)) dlsym(dl, "TF_Version");

    // TD<decltype(TF_NewGraph)> x;

    if (!this->TF_Version) {
      throw std::runtime_error(std::string{dlerror()});
    }
  }

  // TODO: fuj
  static utf::backend* instance;
  static utf::backend& current() { return *utf::backend::instance; }
};

// TODO: utf.cpp
utf::backend* utf::backend::instance;

#undef UTF_GET_SYM
#undef UTF_STRINGIFY
#undef UTF_STRINGIFY2


#define UTF_DEF_HANDLE(WRAPPER_NAME, LIB_NAME)                                  \
  struct WRAPPER_NAME : public handle<TF_##LIB_NAME, TF_Delete##LIB_NAME> { \
    WRAPPER_NAME() : handle(backend::current().TF_New##LIB_NAME()) {}       \
    explicit WRAPPER_NAME(TF_##LIB_NAME* b) : handle(b) {}                  \
  };

UTF_DEF_HANDLE(buffer, Buffer)
UTF_DEF_HANDLE(session_options, SessionOptions)
UTF_DEF_HANDLE(import_graph_def_options, ImportGraphDefOptions)

#undef UTF_DEF_HANDLE


#define UTF_DATATYPE_WRAP(TYPE, DTYPE)       \
  template <>                               \
  TF_DataType type_to_tf_datatype<TYPE>() { \
    return DTYPE;                           \
  }

template <typename T>
TF_DataType type_to_tf_datatype() {
  throw std::runtime_error("Template specialization must be called.");
}

UTF_DATATYPE_WRAP(float, TF_FLOAT)
UTF_DATATYPE_WRAP(int32_t, TF_INT32)
UTF_DATATYPE_WRAP(int64_t, TF_INT64)

#undef UTF_DATATYPE_WRAP


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
  graph() : handle(backend::current().TF_NewGraph()) {}

  operation get_op_checked(const std::string& name) {
    operation op{backend::current().TF_GraphOperationByName(obj, name.c_str())};

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

    backend::current().TF_GraphImportGraphDef(g.obj, buf.obj, opts.obj, s.obj);
    s.check("Graph import");

    return g;
  }

  TF_Output get_initializer(TF_Operation* op) {
    auto b = backend::current();

    string name = b.TF_OperationName(op);
    string target = name + "/Initializer";

    size_t pos = 0;
    TF_Operation* oper;

    while ((oper = b.TF_GraphNextOperation(obj, &pos)) != nullptr) {
      string other_name = b.TF_OperationName(oper);

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
  backend::current().TF_DeleteSession(sess, s.obj);

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
  auto b = backend::current();

  TF_DataType dtype = type_to_tf_datatype<T>();
  int64_t bytes = items * b.TF_DataTypeSize(dtype);

  TF_Tensor* tensor = b.TF_AllocateTensor(dtype, dims.data(), dims.size(), bytes);

  return tensor;
}

template <typename T>
TF_Tensor* allocate_tensor(vector<int64_t> dims, vector<T> values) {
  // TODO: check and delete
  //    TF_AllocateTensor(TF_FLOAT, dims, 1, dims[0] *
  //    TF_DataTypeSize(TF_FLOAT))

  TF_Tensor* tensor = allocate_uninitialized_tensor<T>(dims);
  auto b = backend::current();

  auto tensor_data = reinterpret_cast<T*>(b.TF_TensorData(tensor));
  std::copy(values.begin(), values.end(), tensor_data);

  return tensor;
}

class tensor : public handle<TF_Tensor, TF_DeleteTensor> {
 public:
  explicit tensor(TF_Tensor* t) : handle(t) {}

  template <typename T>
  T* get_data() {
    return reinterpret_cast<T*>(backend::current().TF_TensorData(obj));
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
      : handle(backend::current().TF_NewSession(g.obj, sopts.obj, s.obj)), g(g) {}

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

    backend::current().TF_SessionRun(obj, nullptr, nullptr, nullptr, 0,
        nullptr, nullptr, 0, ops.data(), target_ops.size(), nullptr,
        status.obj);
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
    // TF_Operation* const target_opers[] = {};

    vector<TF_Tensor*> output_values;
    output_values.reserve(outputs.size());

    backend::current().TF_SessionRun(obj, nullptr, inputs.data(),
        map_tensors(input_values).data(), inputs.size(), outputs.data(),
        output_values.data(), outputs.size(), target_ops.data(),
        target_ops.size(), nullptr, status.obj);

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


  void initialize_variables(std::string weights_fname, graph& g, session& sess, status& s) {
    vector<TF_Output> no_outputs;

    ifstream is(weights_fname);
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

      init_inputs.push_back(g.get_initializer(g.get_op_checked(name).op));
      init_targets.push_back(g.get_op_checked(name + "/Assign").op);

      init_input_tensors.push_back(utf::tensor::create<float>(shape, values));
      cout << "var\t" << var["name"] << "\t" << var["shape"] << " ... " << shape[0] << endl;
    }

    sess.run(init_inputs, init_input_tensors, no_outputs, init_targets, s);

    s.check("Session RUN - init all");
  }


}  // namespace utf

void heap_buf_deallocator(void* data, size_t length) { free(data); }

utf::buffer read_file(const char* fname) {
  FILE* f = fopen(fname, "rb");
  if (!f) {
    std::stringstream ss;
    ss << "File '" << fname << "' does not exist." << std::endl;
    throw std::runtime_error(ss.str());
  }

  fseek(f, 0, SEEK_END);
  auto size = ftell(f);
  fseek(f, 0, SEEK_SET);

  void* data = malloc(size);
  fread(data, 1, size, f);

  TF_Buffer* buf = utf::backend::current().TF_NewBuffer();
  buf->data = data;
  buf->length = size;
  buf->data_deallocator = heap_buf_deallocator;

  return utf::buffer{buf};
}
