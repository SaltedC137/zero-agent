#pragma once

#include "chat.hpp"
#include "imodel.hpp"

#include <memory>
#include <string>
#include <vector>

namespace zato {

// Token-budget context window management (Claude Code style).
// Works for both local llama.cpp (token-accurate) and API (char estimate).
class WindowManager
{
public:
  // budget: token limit (local) or char limit (API)
  // api_mode: use char estimation + model summarization for compaction
  WindowManager(std::shared_ptr<IModel> model,
                int budget,
                bool api_mode,
                const std::string& system_prompt);

  // Auto-manage: compact or trim if over 80% of budget.
  // Returns true if compaction/summarization happened.
  bool manage(std::vector<common_chat_msg>& messages);

  [[nodiscard]] int usage(const std::vector<common_chat_msg>& messages) const;

private:
  void trim(std::vector<common_chat_msg>& messages);
  bool compact(std::vector<common_chat_msg>& messages);

  std::shared_ptr<IModel> model_;
  int budget_;
  bool api_mode_;
  std::string system_prompt_;
  bool compacted_once_ = false; // don't compact on every single call
};

} // namespace zato
