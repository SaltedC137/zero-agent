#pragma once

#include <nlohmann/json.hpp>
#include <string>


#include "chat.hpp"

namespace zato {

using json = nlohmann::json;

class Tool {

public:
  virtual ~Tool() = default;

  virtual common_chat_tool get_definition() const = 0;

  virtual std::string execute(const json& arguments) = 0;

  virtual std::string get_name() const = 0;

};

} // namespace zato
