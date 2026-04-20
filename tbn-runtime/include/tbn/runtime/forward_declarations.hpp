#pragma once

#include <memory>
#include <string>
#include <vector>

namespace tbn {

// Forward declarations
class Tensor;
class TBNModel;
class InferenceSession;

// Model forward declarations
struct ModelNode;
struct ModelGraph;

// Tensor helpers
using TensorPtr = std::shared_ptr<Tensor>;
using ModelPtr = std::shared_ptr<TBNModel>;
using SessionPtr = std::shared_ptr<InferenceSession>;

} // namespace tbn