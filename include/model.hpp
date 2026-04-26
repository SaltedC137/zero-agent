#pragma once

#include <cstdint>
#include <functional>
#include <llama.h>
#include <optional>
#include <string>

namespace zota {
using ResponseCallback = std::function<void(const std::string &)>;

// DEFAULTS MODEL CONFIG
struct ModelConfig {
  float min_p = 0.0f;
  float max_p = 1.0f;

  int top_k = 0;
  float temp = 0.0f;
  uint32_t seed = LLAMA_DEFAULT_SEED;

  std::optional<common_chat_>
};

} // namespace zota
