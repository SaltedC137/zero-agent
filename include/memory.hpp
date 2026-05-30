#pragma once

#include "chat.hpp"
#include "imodel.hpp"


#include <memory>
#include <string>
#include <vector>

namespace zato {

// Hierarchical memory: keeps recent messages verbatim, compresses old
// messages into a running summary to stay within context limits.
class MemoryManager
{
public:
  MemoryManager(std::shared_ptr<IModel> model,
                const std::string& system_prompt,
                size_t max_chars = 48000);

  // Compress old messages into a summary if total chars exceed max_chars.
  // Keeps system prompt + last 6 messages verbatim.
  void compact(std::vector<common_chat_msg>& messages);

private:
  std::shared_ptr<IModel> model_;
  std::string system_prompt_;
  size_t max_chars_;
};

} // namespace zato
