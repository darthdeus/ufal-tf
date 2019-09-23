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

}
