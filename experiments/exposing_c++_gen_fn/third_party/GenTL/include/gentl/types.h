#ifndef GENTL_SELECTIONS_H
#define GENTL_SELECTIONS_H

#include <utility>

namespace gentl {

class SimulateOptions {
  bool precompute_gradient_;

 public:
  SimulateOptions() : precompute_gradient_(false) {}

  SimulateOptions& precompute_gradient(bool value) {
    precompute_gradient_ = value;
    return *this;
  }

  [[nodiscard]] bool precompute_gradient() const {
    return precompute_gradient_;
  }
};

class GenerateOptions {
  bool precompute_gradient_;

 public:
  GenerateOptions() : precompute_gradient_(false) {}

  GenerateOptions& precompute_gradient(bool value) {
    precompute_gradient_ = value;
    return *this;
  }

  [[nodiscard]] bool precompute_gradient() const {
    return precompute_gradient_;
  }
};

class UpdateOptions {
  bool precompute_gradient_;
  bool save_;
  bool assert_deterministic_;
  bool ignore_previous_choices_;

 public:
  UpdateOptions()
      : precompute_gradient_(false),
        save_(false),
        assert_deterministic_(false),
        ignore_previous_choices_{false} {}

  UpdateOptions& precompute_gradient(bool value) {
    precompute_gradient_ = value;
    return *this;
  }

  UpdateOptions& save(bool value = true) {
    save_ = value;
    return *this;
  }

  UpdateOptions& assert_deterministic(bool value) {
    assert_deterministic_ = value;
    return *this;
  }

  UpdateOptions& ignore_previous_choices(bool value) {
    ignore_previous_choices_ = value;
    return *this;
  }

  [[nodiscard]] bool precompute_gradient() const {
    return precompute_gradient_;
  }
  [[nodiscard]] bool save() const { return save_; }
  [[nodiscard]] bool assert_deterministic() const {
    return assert_deterministic_;
  }

  // acts like an in-place generate
  // TODO remove the in-place generate function, and similarly for simulate?
  [[nodiscard]] bool ignore_previous_choices() const {
    return ignore_previous_choices_;
  }
};

}  // namespace gentl

namespace gentl::selection {

struct All {};
constexpr All all{};

struct None {};
constexpr None none{};

}  // namespace gentl::selection

namespace gentl::change {

struct NoChange {};
constexpr NoChange no_change{};

template <typename ValueType>
class UnknownChange {
  ValueType new_value_;

 public:
  explicit UnknownChange(const ValueType& new_value) : new_value_{new_value} {}
  explicit UnknownChange(ValueType&& new_value)
      : new_value_{std::move(new_value)} {}
  const ValueType& new_value() const { return new_value_; }
};

}  // namespace gentl::change

#endif  // GENTL_SELECTIONS_H
