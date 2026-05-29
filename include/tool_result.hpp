/**
 * @file model.hpp
 * @author learn from agent.cpp (saltedc137@gmail.com)
 * @brief
 * @version 0.1
 * @date 2026-05-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <exception>
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

class ToolResult
{
private:
  std::variant<std::string, ToolFailure> value_;

public:
  ToolResult(std::string output)
    : value_(std::move(output))
  {
  }

  ToolResult(const char* output)
    : value_(std::string(output))
  {
  }

  ToolResult(ToolFailure err)
    : value_(std::move(err))
  {
  }

  static ToolResult from_exception(const std::exception& e)
  {
    return ToolResult(ToolFailure(e));
  }

  [[nodiscard]] bool has_error() const
  {
    return std::holds_alternative<ToolFailure>(value_);
  }

  [[nodiscard]] bool is_ok() const
  {
    return std::holds_alternative<std::string>(value_);
  }

  [[nodiscard]] const ToolFailure& error() const
  {
    return std::get<ToolFailure>(value_);
  }

  [[nodiscard]] const std::string& output() const
  {
    return std::get<std::string>(value_);
  }

  [[nodiscard]] std::string& output() { return std::get<std::string>(value_); }

  void recover(std::string recovery_message)
  {
    value_ = std::move(recovery_message);
  }
};

} // namespace zato

