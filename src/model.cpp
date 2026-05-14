#include "model.hpp"
#include "chat.hpp"
#include "error.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <ggml-backend.h>
#include <llama.h>
#include <memory>
#include <string>
#include <vector>

// class model weight

namespace zota {

std::shared_ptr<ModelWeight>
ModelWeight::create(const std::string &model_path) {
  std ::shared_ptr<ModelWeight> weight(new ModelWeight());
  // load model
  ggml_backend_load_all();



  llama_model_params model_params = llama_model_default_params();

  model_params.n_gpu_layers = 999; // Use CPU for all layers by default
  model_params.main_gpu = 0; // Use GPU 0 for offloading if needed

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

std::shared_ptr<Model>
Model::create_with_weight(std::shared_ptr<ModelWeight> weight,
                          const ModelConfig &model_config) {
  std::shared_ptr<Model> model(new Model());
  model->weight_ = std::move(weight);
  model->initialize_context(model_config);
  return model;
}

Model::~Model() {
  if (sampler_ != nullptr) {
    llama_sampler_free(sampler_);
  }
  if (context_ != nullptr) {
    llama_free(context_);
  }
  // weights_ is automatically released when ref count drops to zero
}

Model::Model(Model &&other) noexcept
    : weight_(std::move(other.weight_)), context_(other.context_),
      sampler_(other.sampler_),
      processed_tokens_(std::move(other.processed_tokens_)),
      n_past_(other.n_past_), config_(other.config_) {

  other.context_ = nullptr;
  other.sampler_ = nullptr;
  other.n_past_ = 0;
}

Model &Model::operator=(Model &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  if (sampler_ != nullptr) {
    llama_sampler_free(sampler_);
    sampler_ = nullptr;
  }
  if (context_ != nullptr) {
    llama_free(context_);
    context_ = nullptr;
  }

  weight_ = std::move(other.weight_);
  templates_ = other.templates_;
  context_ = other.context_;
  sampler_ = other.sampler_;
  processed_tokens_ = std::move(other.processed_tokens_);
  n_past_ = other.n_past_;
  config_ = other.config_;

  other.templates_ = nullptr;
  other.context_ = nullptr;
  other.sampler_ = nullptr;
  other.n_past_ = 0;

  return *this;
}

void Model::initialize_context(const ModelConfig &model_config) {
  if (!weight_ || weight_->get_model() == nullptr) {
    throw ModelError("Model weight is not initialized");
  }

  config_ = model_config;
  templates_ = weight_->get_templates();

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = static_cast<uint32_t>(std::max(1, model_config.n_ctx));
  ctx_params.n_batch = static_cast<uint32_t>(
      model_config.n_batch > 0 ? model_config.n_batch
                               : std::min(model_config.n_ctx, 1024));
  ctx_params.n_threads = std::max(1, model_config.n_threads);
  ctx_params.n_threads_batch = std::max(1, model_config.n_threads_batch);
  ctx_params.type_k = model_config.cache_type_k;
  ctx_params.type_v = model_config.cache_type_v;

  context_ = llama_init_from_model(weight_->get_model(), ctx_params);
  if (context_ == nullptr) {
    throw ModelError("Failed to initialize llama context");
  }

  auto sampler_params = llama_sampler_chain_default_params();
  sampler_ = llama_sampler_chain_init(sampler_params);
  if (sampler_ == nullptr) {
    throw ModelError("Failed to initialize sampler chain");
  }

  // Minimal sampler stack.
  llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(config_.top_k));
  llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(config_.max_p, 1));
  llama_sampler_chain_add(sampler_, llama_sampler_init_min_p(config_.min_p, 1));
  llama_sampler_chain_add(sampler_,
                          llama_sampler_init_penalties(128, 1.20f, 0.2f, 0.2f));
  llama_sampler_chain_add(sampler_, llama_sampler_init_temp(config_.temp));
  llama_sampler_chain_add(sampler_, llama_sampler_init_dist(config_.seed));
}

std::vector<llama_token> Model::tokenize(const std::string &prompt) const {
  if (prompt.empty()) {
    return {};
  }

  const llama_vocab *vocab = get_vocab();
  if (vocab == nullptr) {
    throw ModelError("Vocabulary is not available");
  }

  int32_t required = llama_tokenize(vocab, prompt.c_str(),
                                    static_cast<int32_t>(prompt.size()), nullptr,
                                    0, true, true);
  if (required == INT32_MIN) {
    throw ModelError("Tokenization overflow");
  }
  if (required < 0) {
    required = -required;
  }

  std::vector<llama_token> tokens(static_cast<size_t>(required));
  int32_t actual = llama_tokenize(vocab, prompt.c_str(),
                                  static_cast<int32_t>(prompt.size()),
                                  tokens.data(), required, true, true);
  if (actual < 0) {
    throw ModelError("Failed to tokenize prompt");
  }

  tokens.resize(static_cast<size_t>(actual));
  return tokens;
}

std::string
Model::generate_from_token(const std::vector<llama_token> &all_tokens,
                           ResponseCallback callback) {
  if (context_ == nullptr || sampler_ == nullptr) {
    throw ModelError("Model context is not initialized");
  }

  const llama_vocab *vocab = get_vocab();
  if (vocab == nullptr) {
    throw ModelError("Vocabulary is not available");
  }

  auto token_to_piece = [&](llama_token token) -> std::string {
    int32_t capacity = 64;
    std::vector<char> buffer(static_cast<size_t>(capacity));

    while (true) {
      int32_t written =
          llama_token_to_piece(vocab, token, buffer.data(), capacity, 0, true);
      if (written >= 0) {
        return std::string(buffer.data(), static_cast<size_t>(written));
      }
      if (written == INT32_MIN) {
        throw ModelError("token_to_piece overflow");
      }
      capacity = std::max(capacity * 2, -written);
      buffer.resize(static_cast<size_t>(capacity));
    }
  };

  llama_memory_clear(llama_get_memory(context_), true);
  llama_sampler_reset(sampler_);
  processed_tokens_.clear();
  n_past_ = 0;

  if (!all_tokens.empty()) {
    std::vector<llama_token> prompt_tokens = all_tokens;
    auto prompt_batch = llama_batch_get_one(
        prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
    if (llama_decode(context_, prompt_batch) != 0) {
      throw ModelError("Failed to decode prompt tokens");
    }
    set_cache(prompt_tokens);
  }

  constexpr int max_new_tokens = 4096;
  constexpr size_t max_output_chars = 12000;
  const std::vector<std::string> stop_sequences = {
      "<|im_end|>",
      "<|endoftext|>",
      "[END_OF_TEXT]",
      "\n**Created Question**",
      "\nCreated Question"};
  std::string output;
  output.reserve(2048);

  for (int i = 0; i < max_new_tokens; ++i) {
    const llama_token token = llama_sampler_sample(sampler_, context_, -1);
    llama_sampler_accept(sampler_, token);

    if (llama_vocab_is_eog(vocab, token)) {
      break;
    }

    std::string piece = token_to_piece(token);
    size_t previous_size = output.size();
    output += piece;

    size_t stop_pos = std::string::npos;
    for (const auto &stop : stop_sequences) {
      size_t pos = output.find(stop);
      if (pos != std::string::npos) {
        stop_pos = stop_pos == std::string::npos ? pos : std::min(stop_pos, pos);
      }
    }

    if (stop_pos != std::string::npos) {
      std::string final_output = output.substr(0, stop_pos);
      if (callback && final_output.size() > previous_size) {
        callback(final_output.substr(previous_size));
      }
      output = final_output;
      break;
    }

    if (callback) {
      callback(piece);
    }

    if (output.size() >= max_output_chars) {
      break;
    }

    processed_tokens_.push_back(token);
    n_past_ = static_cast<int>(processed_tokens_.size());

    llama_token next = token;
    auto next_batch = llama_batch_get_one(&next, 1);
    if (llama_decode(context_, next_batch) != 0) {
      throw ModelError("Failed while decoding generated token");
    }
  }

  return output;
}

common_chat_msg Model::generate(const std::vector<common_chat_msg> &messages,
                                const std::vector<common_chat_tool> & /*tools*/,
                                ResponseCallback callback) {
  if (messages.empty()) {
    return make_assistant_msg("");
  }

  std::string prompt =
      templates_ != nullptr ? templates_->apply(messages) : format_chatml(messages);
  auto tokens = tokenize(prompt);
  auto text = generate_from_token(tokens, callback);
  return make_assistant_msg(text);
}

bool Model::save_cache(const std::string &cache_path) {
  if (context_ == nullptr) {
    throw ModelError("Cannot save cache: context is null");
  }

  return llama_state_save_file(context_, cache_path.c_str(),
                               processed_tokens_.data(),
                               processed_tokens_.size());
}

std::vector<llama_token> Model::load_cache(const std::string &cache_path) {
  if (context_ == nullptr) {
    throw ModelError("Cannot load cache: context is null");
  }

  size_t token_capacity =
      static_cast<size_t>(std::max(4096, std::max(1, config_.n_ctx) * 4));
  std::vector<llama_token> tokens(token_capacity);
  size_t token_count = 0;

  bool ok = llama_state_load_file(context_, cache_path.c_str(), tokens.data(),
                                  tokens.size(), &token_count);
  if (!ok) {
    throw ModelError("Failed to load cache from path: " + cache_path);
  }

  tokens.resize(token_count);
  set_cache(tokens);
  return tokens;
}
} // namespace zota
