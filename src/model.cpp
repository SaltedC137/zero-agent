

#include "chat.hpp"
#include "error.hpp"

#include <algorithm>
#include <ggml-backend.h>
#include <llama.h>
#include <memory>
#include <model.hpp>

// class model weight

namespace zota {

std::shared_ptr<ModelWeight>
ModelWeight::create(const std::string &model_path) {
  std ::shared_ptr<ModelWeight> weight(new ModelWeight());
  // load model
  ggml_backend_load_all();

  llama_model_params model_params = llama_model_default_params();
  weight->model_ = llama_model_load_from_file(model_path.c_str(), model_params);

  if (weight->model_ == nullptr) {
    throw ModelError("Failed to load model from path: " + model_path);
  }

  auto tmpls = common_chat_templates_init(weight->model_);

  if (tmpls.name.empty()) {
    throw ModelError("Failed to initialize chat templates for model: " +
                     model_path);
  }
  weight->templates_ = std::make_shared<common_chat_templates>(tmpls);

  return weight;
}

ModelWeight::~ModelWeight() {
  if (model_ != nullptr) {
    llama_model_free(model_);
    model_ = nullptr;
  }
}

std::shared_ptr<Model> Model::create(const std::string &model_path,
                                     const ModelConfig &config) {
  auto weights = ModelWeight::create(model_path);
  return create_with_weight(std::move(weights), config);
}

std::shared_ptr<Model>Model::create_with_weight(std::shared_ptr<ModelWeight>)

} // namespace zota