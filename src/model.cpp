/**
 * @file model.cpp
 * @author Aska Lyn (saltedc137@gmail)
 * @brief Model implementation — GGUF loading, token generation, tool call
 * parsing.
 * @version 0.1
 * @date 2026-04-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "model.hpp"
#include "chat.hpp"
#include "error.hpp"
#include "tool.hpp"

#include <algorithm>
#include <cstdint>
#include <ggml-backend.h>
#include <llama.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// class model weight

namespace zato {

static std::string
strip_code_fence_json(const std::string& text)
{
  std::string t = trim_copy(text);
  if (t.rfind("```", 0) != 0) {
    return t;
  }
  const auto first_nl = t.find('\n');
  if (first_nl == std::string::npos) {
    return t;
  }
  const auto last_fence = t.rfind("```");
  if (last_fence == std::string::npos || last_fence <= first_nl) {
    return t;
  }
  return trim_copy(t.substr(first_nl + 1, last_fence - first_nl - 1));
}

std::optional<common_chat_msg>
try_parse_tool_call_message(const std::string& text)
{
  std::string t = strip_code_fence_json(text);
  t = trim_copy(t);

  if (t.empty()) {
    return std::nullopt;
  }

  // The model may output <think>...</think> or other text before the JSON.
  // Search for the tool_calls marker and parse from there.
  if (t.front() != '{') {
    const auto marker = t.find("\"tool_calls\"");
    if (marker == std::string::npos) { return std::nullopt;
}
    // Walk back to the opening brace
    auto brace = t.rfind('{', marker);
    if (brace == std::string::npos) { return std::nullopt;
}
    t = t.substr(brace);
    t = trim_copy(t);
  }

  try {
    json j = json::parse(t);
    if (!j.is_object() || !j.contains("tool_calls") ||
        !j["tool_calls"].is_array()) {
      return std::nullopt;
    }

    common_chat_msg msg;
    msg.role = MessageRole::ASSISTANT;

    if (j.contains("content") && j["content"].is_string()) {
      msg.content = j["content"].get<std::string>();
    }

    const auto& calls = j["tool_calls"];
    msg.tool_calls.reserve(calls.size());

    for (size_t i = 0; i < calls.size(); ++i) {
      const auto& c = calls.at(i);
      if (!c.is_object() || !c.contains("tool_name")) {
        continue;
      }

      common_chat_tool_call call;
      call.tool_name = c.at("tool_name").get<std::string>();

      if (c.contains("tool_args")) {
        const auto& args = c.at("tool_args");
        call.tool_args =
          args.is_string() ? args.get<std::string>() : args.dump();
      } else {
        call.tool_args = "{}";
      }

      if (c.contains("tool_call_id") && c.at("tool_call_id").is_string()) {
        call.tool_call_id = c.at("tool_call_id").get<std::string>();
      } else {
        call.tool_call_id = "call_" + std::to_string(i);
      }

      msg.tool_calls.push_back(std::move(call));
    }

    if (msg.tool_calls.empty()) {
      return std::nullopt;
    }

    return msg;
  } catch (...) {
    return std::nullopt;
  }
}

std::shared_ptr<ModelWeight>
ModelWeight::create(const std::string& model_path, int n_gpu_layers)
{
  std ::shared_ptr<ModelWeight> weight(new ModelWeight());

  llama_backend_init();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = n_gpu_layers;

  // MoE expert tensors stay on CPU (like llama-cli --cpu-moe).
  // Without this, each "layer" includes massive expert FFN weights
  // that can't fit in VRAM alongside attention layers.
  static constexpr size_t kMaxOverrides = 8;
  static llama_model_tensor_buft_override overrides[kMaxOverrides + 1];
  if (n_gpu_layers > 0) {
    auto *const cpu_buft = ggml_backend_cpu_buffer_type();
    const char* moe_patterns[] = {
      ".*ffn_gate_exps\\.weight", ".*ffn_up_exps\\.weight",
      ".*ffn_down_exps\\.weight", ".*ffn_gate_inp\\.weight",
      ".*ffn_gate_shexp\\.weight", ".*ffn_up_shexp\\.weight",
      ".*ffn_down_shexp\\.weight",
    };
    for (size_t i = 0; i < sizeof(moe_patterns) / sizeof(moe_patterns[0]); ++i) {
      overrides[i] = { moe_patterns[i], cpu_buft };
    }
    overrides[sizeof(moe_patterns) / sizeof(moe_patterns[0])] = { nullptr, nullptr };
    model_params.tensor_buft_overrides = overrides;
  }

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

ModelWeight::~ModelWeight()
{
  if (model_ != nullptr) {
    llama_model_free(model_);
    model_ = nullptr;
  }
}

std::shared_ptr<Model>
Model::create(const std::string& model_path, const ModelConfig& config)
{
  auto weights = ModelWeight::create(model_path, config.n_gpu_layers);
  return create_with_weight(std::move(weights), config);
}

std::shared_ptr<Model>
Model::create_with_weight(std::shared_ptr<ModelWeight> weight,
                          const ModelConfig& model_config)
{
  std::shared_ptr<Model> model(new Model());
  model->weight_ = std::move(weight);
  model->initialize_context(model_config);
  return model;
}

Model::~Model()
{
  if (sampler_ != nullptr) {
    llama_sampler_free(sampler_);
  }
  if (context_ != nullptr) {
    llama_free(context_);
  }
  // weights_ is automatically released when ref count drops to zero
}

Model::Model(Model&& other) noexcept
  : weight_(std::move(other.weight_))
  , context_(other.context_)
  , sampler_(other.sampler_)
  , processed_tokens_(std::move(other.processed_tokens_))
  , n_past_(other.n_past_)
  , config_(other.config_)
  , cached_tool_instruction_(std::move(other.cached_tool_instruction_))
  , cached_tools_(std::move(other.cached_tools_))
{

  other.context_ = nullptr;
  other.sampler_ = nullptr;
  other.n_past_ = 0;
}

Model&
Model::operator=(Model&& other) noexcept
{
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
  context_ = other.context_;
  sampler_ = other.sampler_;
  processed_tokens_ = std::move(other.processed_tokens_);
  n_past_ = other.n_past_;
  config_ = other.config_;
  cached_tool_instruction_ = std::move(other.cached_tool_instruction_);
  cached_tools_ = std::move(other.cached_tools_);

  other.context_ = nullptr;
  other.sampler_ = nullptr;
  other.n_past_ = 0;

  return *this;
}

void
Model::initialize_context(const ModelConfig& model_config)
{
  if (!weight_ || weight_->get_model() == nullptr) {
    throw ModelError("Model weight is not initialized");
  }

  config_ = model_config;

  const bool has_gpu = model_config.n_gpu_layers > 0;

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = static_cast<uint32_t>(std::max(1, model_config.n_ctx));
  const uint32_t batch = static_cast<uint32_t>(
    model_config.n_batch > 0 ? model_config.n_batch
                             : std::min(model_config.n_ctx, 1024));
  ctx_params.n_batch = batch;
  ctx_params.n_ubatch = batch;
  ctx_params.n_threads = std::max(1, model_config.n_threads);
  ctx_params.n_threads_batch = std::max(1, model_config.n_threads_batch);
  ctx_params.type_k = model_config.cache_type_k;
  ctx_params.type_v = model_config.cache_type_v;
  ctx_params.flash_attn_type = model_config.flash_attn_type;
  ctx_params.offload_kqv = has_gpu && model_config.offload_kqv;
  ctx_params.op_offload = has_gpu;
  ctx_params.no_perf = true;

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

std::vector<llama_token>
Model::tokenize(const std::string& prompt) const
{
  if (prompt.empty()) {
    return {};
  }

  const llama_vocab* vocab = get_vocab();
  if (vocab == nullptr) {
    throw ModelError("Vocabulary is not available");
  }

  int32_t required = llama_tokenize(vocab,
                                    prompt.c_str(),
                                    static_cast<int32_t>(prompt.size()),
                                    nullptr,
                                    0,
                                    true,
                                    true);
  if (required == INT32_MIN) {
    throw ModelError("Tokenization overflow");
  }
  if (required < 0) {
    required = -required;
  }

  std::vector<llama_token> tokens(static_cast<size_t>(required));
  int32_t actual = llama_tokenize(vocab,
                                  prompt.c_str(),
                                  static_cast<int32_t>(prompt.size()),
                                  tokens.data(),
                                  required,
                                  true,
                                  true);
  if (actual < 0) {
    throw ModelError("Failed to tokenize prompt");
  }

  tokens.resize(static_cast<size_t>(actual));
  return tokens;
}

std::string
Model::generate_from_token(const std::vector<llama_token>& all_tokens,
                           ResponseCallback callback)
{
  if (context_ == nullptr || sampler_ == nullptr) {
    throw ModelError("Model context is not initialized");
  }

  const llama_vocab* vocab = get_vocab();
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

  llama_sampler_reset(sampler_);

  // Find common prefix with cached tokens to reuse KV cache across turns
  size_t common_prefix = 0;
  if (!processed_tokens_.empty() && !all_tokens.empty()) {
    const size_t min_len =
      std::min(processed_tokens_.size(), all_tokens.size());
    while (common_prefix < min_len &&
           processed_tokens_[common_prefix] == all_tokens[common_prefix]) {
      ++common_prefix;
    }
  }

  // Trim stale KV cache entries beyond the common prefix
  if (common_prefix < processed_tokens_.size()) {
    llama_memory_seq_rm(
      llama_get_memory(context_), 0, static_cast<llama_pos>(common_prefix), -1);
    processed_tokens_.resize(common_prefix);
    n_past_ = static_cast<int32_t>(common_prefix);
  }

  // Decode new suffix tokens (or all tokens on first call)
  if (common_prefix < all_tokens.size()) {
    const auto suffix_begin =
      all_tokens.begin() + static_cast<std::ptrdiff_t>(common_prefix);
    std::vector<llama_token> new_tokens(suffix_begin, all_tokens.end());

    for (size_t i = 0; i < new_tokens.size();) {
      const size_t batch_size =
        std::min(static_cast<size_t>(config_.n_batch), new_tokens.size() - i);
      auto batch = llama_batch_get_one(new_tokens.data() + i,
                                       static_cast<int32_t>(batch_size));
      if (llama_decode(context_, batch) != 0) {
        throw ModelError("Failed to decode prompt tokens");
      }
      i += batch_size;
    }

    processed_tokens_.insert(
      processed_tokens_.end(), new_tokens.begin(), new_tokens.end());
    n_past_ = static_cast<int32_t>(processed_tokens_.size());
  }

  constexpr int max_new_tokens = 2048;
  constexpr size_t max_output_chars = 8000;
  const std::vector<std::string> stop_sequences = { "<|im_end|>",
                                                    "<|endoftext|>",
                                                    "[END_OF_TEXT]",
                                                    "\n**Created Question**",
                                                    "\nCreated Question" };
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
    for (const auto& stop : stop_sequences) {
      size_t pos = output.find(stop);
      if (pos != std::string::npos) {
        stop_pos =
          stop_pos == std::string::npos ? pos : std::min(stop_pos, pos);
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

common_chat_msg
Model::generate(const std::vector<common_chat_msg>& messages,
                const std::vector<common_chat_tool>& tools,
                ResponseCallback callback)
{
  if (messages.empty()) {
    return make_assistant_msg("");
  }

  (void)tools; // tool calling rules are in the system prompt

  // Build renderable messages
  std::vector<common_chat_msg> renderable = messages;

  std::string prompt;

  // Try model's built-in chat template first
  const llama_model* model =
    weight_ != nullptr ? weight_->get_model() : nullptr;
  if (model != nullptr) {
    int tmpl_len =
      llama_model_meta_val_str(model, "tokenizer.chat_template", nullptr, 0);
    if (tmpl_len > 0) {
      std::string tmpl(static_cast<size_t>(tmpl_len), '\0');
      llama_model_meta_val_str(
        model, "tokenizer.chat_template", tmpl.data(), tmpl_len);

      std::vector<llama_chat_message> chat_messages;
      chat_messages.reserve(renderable.size());
      for (const auto& msg : renderable) {
        chat_messages.push_back(
          { role_to_string(msg.role).c_str(), msg.content.c_str() });
      }

      static constexpr int kMaxTemplateRetries = 3;
      int32_t needed = llama_chat_apply_template(tmpl.c_str(),
                                                 chat_messages.data(),
                                                 chat_messages.size(),
                                                 true,
                                                 nullptr,
                                                 0);

      for (int attempt = 0; needed > 0 && attempt < kMaxTemplateRetries;
           ++attempt) {
        std::string rendered(static_cast<size_t>(needed) + 1, '\0');
        int32_t written =
          llama_chat_apply_template(tmpl.c_str(),
                                    chat_messages.data(),
                                    chat_messages.size(),
                                    true,
                                    rendered.data(),
                                    static_cast<int32_t>(rendered.size()));

        if (written < 0) {
          break;
        }

        if (written > static_cast<int32_t>(rendered.size())) {
          needed = written;
          continue;
        }

        rendered.resize(static_cast<size_t>(written));
        if (!rendered.empty() && rendered.back() == '\0') {
          rendered.pop_back();
        }
        prompt = std::move(rendered);
        break;
      }
    }
  }

  // Fallback: manual chatml formatting
  if (prompt.empty()) {
    auto* tmpl = weight_ ? weight_->get_templates() : nullptr;
    prompt =
      tmpl != nullptr ? tmpl->apply(renderable) : format_chatml(renderable);
  }

  auto tokens = tokenize(prompt);
  auto text = generate_from_token(tokens, callback);

  if (auto parsed = try_parse_tool_call_message(text)) {
    return *parsed;
  }

  return make_assistant_msg(text);
}

bool
Model::save_cache(const std::string& cache_path)
{
  if (context_ == nullptr) {
    throw ModelError("Cannot save cache: context is null");
  }

  return llama_state_save_file(context_,
                               cache_path.c_str(),
                               processed_tokens_.data(),
                               processed_tokens_.size());
}

std::vector<llama_token>
Model::load_cache(const std::string& cache_path)
{
  if (context_ == nullptr) {
    throw ModelError("Cannot load cache: context is null");
  }

  size_t token_capacity =
    static_cast<size_t>(std::max(4096, std::max(1, config_.n_ctx) * 4));
  std::vector<llama_token> tokens(token_capacity);
  size_t token_count = 0;

  bool ok = llama_state_load_file(
    context_, cache_path.c_str(), tokens.data(), tokens.size(), &token_count);
  if (!ok) {
    throw ModelError("Failed to load cache from path: " + cache_path);
  }

  tokens.resize(token_count);
  set_cache(tokens);
  return tokens;
}
} // namespace zato
