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

#include "chat.hpp"

#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace zato {

using json = nlohmann::json;

class Tool
{
public:
  virtual ~Tool() = default;

  virtual common_chat_tool get_definition() const = 0;
  virtual std::string execute(const json& arguments) = 0;
  virtual std::string get_name() const = 0;
};

// Runtime tool registry.
//
// This lets you register tools by name (factory functions) and create them at
// runtime, e.g. when loading tool configs.
class ToolRegistry
{
public:
  using Factory = std::function<std::unique_ptr<Tool>()>;

  // Registers a factory for a tool name.
  // Returns false if the name already exists.
  static bool register_factory(const std::string& name, Factory factory);

  // Creates a tool instance by name. Returns nullptr if not found.
  static std::unique_ptr<Tool> create(const std::string& name);

  // Lists all registered tool names.
  static std::vector<std::string> list_names();

  // Builds tool definitions for all registered tools.
  static std::vector<common_chat_tool> list_definitions();
};

// Convenience macro for static registration.
//
// Usage:
//   class MyTool : public zato::Tool { ... };
//   ZATO_REGISTER_TOOL(MyTool, "my_tool");
#define ZATO_REGISTER_TOOL(ToolType, ToolName)                                \
  namespace {                                                                  \
  const bool g_zato_tool_registered_##ToolType =                               \
    ::zato::ToolRegistry::register_factory(                                    \
      ToolName, []() { return std::make_unique<ToolType>(); });               \
  }

} // namespace zato
