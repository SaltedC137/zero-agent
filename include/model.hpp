/**
 * @file model.hpp
 * @author learn from agent.cpp (saltedc137@gmail.com)
 * @brief
 * @version 0.1
 * @date 2026-04-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "chat.hpp"

#include <cstdint>
#include <functional>
#include <ggml.h>
#include <llama.h>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace zota {
using ResponseCallback = std::function<void(const std::string &)>;

// DEFAULTS MODEL CONFIG
struct ModelConfig {
  float min_p = 0.0f;
  float max_p = 1.0f;

  int top_k = 0;
  float temp = 0.0f;
  uint32_t seed = LLAMA_DEFAULT_SEED;

  std::optional<common_chat_format> chat_format = std::nullopt;
  int n_ctx = 4096;
  int n_batch = -1;
  int n_threads =
      static_cast<int>(std::max(1u, std::thread::hardware_concurrency() - 1));
  int n_threads_batch =
      static_cast<int>(std::max(1u, std::thread::hardware_concurrency() - 1));

  ggml_type cache_type_k = GGML_TYPE_F16;
  ggml_type cache_type_v = GGML_TYPE_F16;
};

class Model;

class ModelWeight {
  friend class Model;
  

public:
  static std::shared_ptr<ModelWeight> create(const std::string &model_path);

  ~ModelWeight();

  ModelWeight(const ModelWeight &) = delete;
  ModelWeight &operator=(const ModelWeight &) = delete;
  ModelWeight(ModelWeight &&) = delete;
  ModelWeight &operator=(ModelWeight &&) = delete;

  [[nodiscard]] llama_model *get_model() const { return model_; }

  [[nodiscard]] common_chat_templates *get_templates() const {
    return templates_.get();
  }

  [[nodiscard]] const llama_vocab *get_vocab() const {
    if (model_ == nullptr) {
      return nullptr;
    }
    return llama_model_get_vocab(model_);
  }

private:
  ModelWeight() = default;
  llama_model *model_ = nullptr;
  std::shared_ptr<common_chat_templates> templates_;
};

// class model

class Model {

public:
  /// @brief Initialize GGUF file
  /// @param model_path The path to the GGUF file
  /// @param config The model configuration

  static std::shared_ptr<Model> create(const std::string &model_path,
                                       const ModelConfig &config);

  /// @brief Create a new model instance
  /// @param model_path The path to the GGUF file
  /// @param config The model configuration

  static std::shared_ptr<Model>
      create_with_weight(std::shared_ptr<ModelWeight>);

  ~Model();

  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;
  // move operations
  Model(Model &&other) noexcept;
  Model &operator=(Model &&other) noexcept;

  // Generate a response based on the input messages and tools
  common_chat_msg generate(const std::vector<common_chat_msg> &messages,
                           const std::vector<common_chat_tool> &tools,
                           ResponseCallback callback = nullptr);

  // Generate a response based on the input messages and tools, with a custom
  std::string generate_from_token(const std::vector<llama_token> &all_tokens,
                                  ResponseCallback callback = nullptr);

  // tokenized
  std::vector<llama_token> tokenize(const std::string &prompt) const;

  [[nodiscard]] common_chat_templates *get_templates() const {
    return weight_->get_templates();
  }

  [[nodiscard]] const llama_vocab *get_vocab() const {
    return weight_->get_vocab();
  }

  [[nodiscard]] llama_context *get_context() const { return context_; }

  [[nodiscard]] std::shared_ptr<ModelWeight> get_weight() const {
    return weight_;
  }

  bool save_cache(const std::string &cache_path);

  std::vector<llama_token> load_cache(const std::string &cache_path);

private:
  void set_cache(const std::vector<llama_token> &tokens) {
    processed_tokens_ = tokens;
    n_past_ = static_cast<int>(tokens.size());
  }
  Model() = default;

  void initialize_context(const ModelConfig &model_config);

  std::shared_ptr<ModelWeight> weight_;
  common_chat_templates *templates_ = nullptr;
  llama_context *context_ = nullptr;
  int n_past_ = 0;
  std::vector<llama_token> processed_tokens_;
  ModelConfig config_;
};

} // namespace zota
