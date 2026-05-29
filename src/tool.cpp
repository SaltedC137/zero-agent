#include "tool.hpp"
#include "chat.hpp"
#include "error.hpp"
#include "model.hpp"
#include <fstream>

namespace {

class EchoTool final : public zato::Tool
{
public:
  zato::common_chat_tool get_definition() const override
  {
    return { .name = "echo",
             .description = "Echo back the provided text.",
             .params = { { .name = "text",
                           .type = "string",
                           .description = "Text to echo.",
                           .required = true } } };
  }

  std::string execute(const zato::json& arguments) override
  {
    if (!arguments.contains("text") || !arguments.at("text").is_string()) {
      throw zato::ToolArgumentError("echo", "'text' must be a string");
    }
    return arguments.at("text").get<std::string>();
  }

  std::string get_name() const override { return "echo"; }
};

class AddTool final : public zato::Tool
{
public:
  zato::common_chat_tool get_definition() const override
  {
    return { .name = "add",
             .description = "Add two integers a and b.",
             .params = { { .name = "a",
                           .type = "integer",
                           .description = "First integer.",
                           .required = true },
                         { .name = "b",
                           .type = "integer",
                           .description = "Second integer.",
                           .required = true } } };
  }

  std::string execute(const zato::json& arguments) override
  {
    if (!arguments.contains("a") || !arguments.contains("b")) {
      throw zato::ToolArgumentError("add", "'a' and 'b' are required");
    }
    if (!arguments.at("a").is_number_integer() ||
        !arguments.at("b").is_number_integer()) {
      throw zato::ToolArgumentError("add", "'a' and 'b' must be integers");
    }
    const long long a = arguments.at("a").get<long long>();
    const long long b = arguments.at("b").get<long long>();
    return std::to_string(a + b);
  }

  std::string get_name() const override { return "add"; }
};

// Trim leading and trailing whitespace from a string.
std::string
trim(const std::string& text)
{
  const auto begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }

  const auto end = text.find_last_not_of(" \t\r\n");
  return text.substr(begin, end - begin + 1);
}

std::string
load_system_prompt_file(const std::string& path)
{
  std::ifstream file(path);
  if (!file.is_open()) {
    throw zato::Error("Failed to open system prompt file: " + path);
  }

  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  content = trim(content);

  const std::size_t begin = content.find("~~~");
  const std::size_t end = content.rfind("~~~");
  if (begin != std::string::npos && end != std::string::npos && end > begin) {
    const std::size_t line_begin = content.find('\n', begin);
    if (line_begin != std::string::npos) {
      return trim(content.substr(line_begin + 1, end - line_begin - 1));
    }
  }

  return content;

  // Safe demo tool: reads a small chunk of a *relative* text file.
  // - Rejects absolute paths and '..' path traversal
  // - Limits output size via max_bytes
  class ReadTextFileTool final : public zato::Tool
  {
  public:
    zato::common_chat_tool get_definition() const override
    {
      return { .name = "read_text_file",
               .description = "Read a UTF-8 text file from a relative path and "
                              "return at most max_bytes bytes.",
               .params = {
                 { .name = "path",
                   .type = "string",
                   .description =
                     "Relative file path, e.g. 'prompt/system.md'.",
                   .required = true },
                 { .name = "max_bytes",
                   .type = "integer",
                   .description =
                     "Maximum bytes to read (1..20000). Default 4000.",
                   .required = false } } };
    }

    std::string execute(const zato::json& arguments) override
    {
      const std::string tool_name = get_name();

      if (!arguments.contains("path") || !arguments.at("path").is_string()) {
        throw zato::ToolArgumentError(tool_name, "'path' must be a string");
      }

      const std::string path = arguments.at("path").get<std::string>();
      if (path.empty()) {
        throw zato::ToolArgumentError(tool_name, "'path' must not be empty");
      }
      if (path.rfind("/", 0) == 0) {
        throw zato::ToolArgumentError(tool_name,
                                      "absolute paths are not allowed");
      }
      if (path.find("..") != std::string::npos) {
        throw zato::ToolArgumentError(tool_name,
                                      "'..' path traversal is not allowed");
      }

      std::size_t max_bytes = 4000;
      if (arguments.contains("max_bytes")) {
        if (!arguments.at("max_bytes").is_number_integer()) {
          throw zato::ToolArgumentError(tool_name,
                                        "'max_bytes' must be an integer");
        }
        const long long v = arguments.at("max_bytes").get<long long>();
        if (v < 1 || v > 20000) {
          throw zato::ToolArgumentError(
            tool_name, "'max_bytes' must be in range 1..20000");
        }
        max_bytes = static_cast<std::size_t>(v);
      }

      std::ifstream file(path, std::ios::in | std::ios::binary);
      if (!file.is_open()) {
        throw zato::ToolArgumentError(tool_name,
                                      "failed to open file: " + path);
      }

      std::string content;
      content.resize(max_bytes);
      file.read(content.data(), static_cast<std::streamsize>(max_bytes));
      const std::streamsize n = file.gcount();
      if (n < 0) {
        throw zato::ToolArgumentError(tool_name,
                                      "failed while reading file: " + path);
      }
      content.resize(static_cast<std::size_t>(n));

      bool truncated = false;
      if (static_cast<std::size_t>(n) == max_bytes) {
        const int next = file.peek();
        truncated = (next != std::char_traits<char>::eof());
      }

      zato::json out;
      out["path"] = path;
      out["max_bytes"] = max_bytes;
      out["truncated"] = truncated;
      out["content"] = content;
      return out.dump();
    }

    std::string get_name() const override { return "read_text_file"; }
  };
}
}