#pragma once

namespace utf {

template <typename Obj, void (*Deleter)(Obj*)>
class handle {
 public:
  Obj* obj = nullptr;

  explicit handle(Obj* o) : obj(o){};
  handle(const handle&) = delete;
  handle(handle&& h) noexcept : obj(h.obj) { h.obj = nullptr; }

  handle& operator=(const handle&) = delete;
  handle& operator=(handle&& h) noexcept {
    this->obj = h.obj;
    h.obj = nullptr;
    return *this;
  }

  ~handle() {
    if (obj) {
//      std::cout << "deleting " << obj << std::endl;
      Deleter(obj);
    }
  }
};

//  template<typename Obj, typename F>
//  class deleter {
//    F f;
//
//    deleter(F f): f(f) {};
//    void operator()(Obj* obj) {
//      f(obj);
//    }
//  };


#define DEF_HANDLE(WRAPPER_NAME, LIB_NAME)                                  \
  struct WRAPPER_NAME : public handle<TF_##LIB_NAME, TF_Delete##LIB_NAME> { \
    WRAPPER_NAME() : handle(TF_New##LIB_NAME()) {}                          \
    explicit WRAPPER_NAME(TF_##LIB_NAME* b) : handle(b) {}                  \
  };

#define TF_DATATYPE_WRAP(TYPE, DTYPE)       \
  template <>                               \
  TF_DataType type_to_tf_datatype<TYPE>() { \
    return DTYPE;                           \
  }

template <typename T>
TF_DataType type_to_tf_datatype() {
  throw std::runtime_error("Template specialization must be called.");
}

TF_DATATYPE_WRAP(float, TF_FLOAT)
TF_DATATYPE_WRAP(int32_t, TF_INT32)
TF_DATATYPE_WRAP(int64_t, TF_INT64)

}
