#pragma once

#include <exception>
#include <stdexcept>
#include <string>
#include <variant>

namespace zato {

struct ToolFailure
{
  std::string message;

  explicit ToolFailure(const std::string& msg)
    : message(msg)
  {
  }

  explicit ToolFailure(const std::exception& e)
    : message(e.what())
  {
  }
};

} // namespace zato