
/**
 * @file model.hpp
 * @author learn from agent.cpp (saltedc137@gmail.com)
 * @brief
 * @version 0.1
 * @date 2026-04-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <exception>
#include <stdexcept>
#include <string>

namespace zota {

// Base error class for all errors in the library
class Error : public std::runtime_error {
public:
  explicit Error(const std::string &msg) : std::runtime_error(msg) {}
};

// Error class for model-related errors
class ModelError : public Error {
public:
  explicit ModelError(const std::string &msg) : Error("Model Error: " + msg) {}
};

// tool call error, which includes the tool name and the error message
class ToolError : public Error {
private:
  std::string tool_name_;

public:
  explicit ToolError(const std::string &tool_name, const std::string &msg)
      : Error("Tool Error: " + tool_name + ": " + msg) {}
  [[nodiscard]] const std::string &get_tool_name() const { return tool_name_; }
};

// tool not find error, which includes the tool name
class ToolNotFoundError : public ToolError {
public:
  explicit ToolNotFoundError(const std::string &tool_name)
      : ToolError(tool_name, "Tool not found") {}
};

// tool argument error, which includes the tool name and the argument error
// message
class ToolArgumentError : public ToolError {
public:
  explicit ToolArgumentError(const std::string &tool_name,
                             const std::string &msg)
      : ToolError(tool_name, "Invalid argument: " + msg) {}
};

// mcp

class McpError : public Error {
public:
  explicit McpError(const std::string &msg) : Error("MCP Error: " + msg) {}
};

class ToolExecutionSkipped : public std::exception {
private:
  std::string message_;

public:
  explicit ToolExecutionSkipped(
      const std::string &msg = "Tool execution skipped")
      : message_(msg) {}

  [[nodiscard]] const char *what() const noexcept override {
    return message_.c_str();
  }

  [[nodiscard]] const std::string &get_message() const { return message_; }
};

} // namespace zota