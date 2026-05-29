#pragma once

#include "chat.hpp"
#include "model.hpp"

#include <memory>
#include <string>
#include <vector>

namespace zato {

// Manages context window: trims old messages when the total token count
// approaches the model's context limit (n_ctx), preserving the system
// message and the most recent conversation turns.
class ContextManager
{
public:
  ContextManager(std::shared_ptr<Model> model,
                 int n_ctx,
                 float trim_threshold = 0.8f)
    : model_(std::move(model))
    , n_ctx_(n_ctx)
    , trim_threshold_(trim_threshold)
  {
  }

  void trim(std::vector<common_chat_msg>& messages) const
  {
    if (messages.empty()) {
      return;
    }

    const int limit = static_cast<int>(n_ctx_ * trim_threshold_);
    int total = count_tokens(messages);
    if (total <= limit) {
      return;
    }

    // Find system message boundary (always keep)
    size_t keep_from = 0;
    if (messages[0].role == MessageRole::SYSTEM) {
      keep_from = 1;
    }

    // Remove oldest non-system messages until we fit, but always keep
    // at least the last 4 messages (user + assistant + tool + reply)
    const size_t min_keep = 4;
    while (keep_from + min_keep < messages.size()) {
      total -= estimate_msg_tokens(messages[keep_from]);
      ++keep_from;
      if (total <= limit)
        break;
    }

    if (keep_from > 0 && messages[0].role == MessageRole::SYSTEM) {
      messages.erase(messages.begin() + 1,
                     messages.begin() + static_cast<std::ptrdiff_t>(keep_from));
    } else if (keep_from > 0) {
      messages.erase(messages.begin(),
                     messages.begin() + static_cast<std::ptrdiff_t>(keep_from));
    }
  }

  [[nodiscard]] int count_tokens(
    const std::vector<common_chat_msg>& messages) const
  {
    return estimate_tokens(format_chatml(messages));
  }

private:
  // Rough per-message estimate without formatting the full prompt
  static int estimate_msg_tokens(const common_chat_msg& msg)
  {
    // Chinese ~2 chars/token, ASCII ~4 chars/token; use 3 as rough average
    int n = static_cast<int>(msg.content.size()) / 3;
    for (const auto& tc : msg.tool_calls) {
      n += static_cast<int>(tc.tool_name.size() + tc.tool_args.size()) / 3 + 4;
    }
    return std::max(1, n);
  }

  int estimate_tokens(const std::string& text) const
  {
    if (auto m = model_.lock()) {
      auto tokens = m->tokenize(text);
      return static_cast<int>(tokens.size());
    }
    return static_cast<int>(text.size() / 3);
  }

  std::weak_ptr<Model> model_;
  int n_ctx_;
  float trim_threshold_;
};

} // namespace zato
