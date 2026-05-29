#pragma once

#pragma once

#include "chat.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace zato {

using ResponseCallback = std::function<void(const std::string&)>;

// Abstract interface for model backends (local llama.cpp or remote API).
struct IModel
{
  virtual ~IModel() = default;

  virtual common_chat_msg generate(
    const std::vector<common_chat_msg>& messages,
    const std::vector<common_chat_tool>& tools,
    ResponseCallback callback = nullptr) = 0;
};

} // namespace zato
