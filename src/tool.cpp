/**
 * @file tool.cpp
 * @author Aska Lyn (saltedc137@gmail)
 * @brief Built-in tool implementations and ToolRegistry thread-safe factory map.
 * @version 0.1
 * @date 2026-04-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "tool.hpp"

#include "error.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sys/wait.h>
#include <unordered_map>

namespace zato {

namespace {

using RegistryMap = std::unordered_map<std::string, ToolRegistry::Factory>;

RegistryMap&
registry_map()
{
  static RegistryMap map;
  return map;
}

std::mutex&
registry_mutex()
{
  static std::mutex m;
  return m;
}

class EchoTool final : public Tool
{
public:
  common_chat_tool get_definition() const override
  {
    return { .name = "echo",
             .description = "Echo back the provided text.",
             .params = { { .name = "text",
                           .type = "string",
                           .description = "Text to echo.",
                           .required = true } } };
  }

  std::string execute(const json& arguments) override
  {
    if (!arguments.contains("text") || !arguments.at("text").is_string()) {
      throw ToolArgumentError(get_name(), "'text' must be a string");
    }
    return arguments.at("text").get<std::string>();
  }

  std::string get_name() const override { return "echo"; }
};

class AddTool final : public Tool
{
public:
  common_chat_tool get_definition() const override
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

  std::string execute(const json& arguments) override
  {
    if (!arguments.contains("a") || !arguments.contains("b")) {
      throw ToolArgumentError(get_name(), "'a' and 'b' are required");
    }
    if (!arguments.at("a").is_number_integer() ||
        !arguments.at("b").is_number_integer()) {
      throw ToolArgumentError(get_name(), "'a' and 'b' must be integers");
    }
    const long long a = arguments.at("a").get<long long>();
    const long long b = arguments.at("b").get<long long>();
    return std::to_string(a + b);
  }

  std::string get_name() const override { return "add"; }
};

class ReadTextFileTool final : public Tool
{
public:
  common_chat_tool get_definition() const override
  {
    return { .name = "read_text_file",
             .description =
               "Read a UTF-8 text file from a relative path and return at most max_bytes bytes.",
             .params = { { .name = "path",
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

  std::string execute(const json& arguments) override
  {
    const std::string tool_name = get_name();

    if (!arguments.contains("path") || !arguments.at("path").is_string()) {
      throw ToolArgumentError(tool_name, "'path' must be a string");
    }

    const std::string path = arguments.at("path").get<std::string>();
    if (path.empty()) {
      throw ToolArgumentError(tool_name, "'path' must not be empty");
    }
    if (path.rfind('/', 0) == 0) {
      throw ToolArgumentError(tool_name, "absolute paths are not allowed");
    }
    if (path.find("..") != std::string::npos) {
      throw ToolArgumentError(tool_name, "'..' path traversal is not allowed");
    }

    std::size_t max_bytes = 4000;
    if (arguments.contains("max_bytes")) {
      if (!arguments.at("max_bytes").is_number_integer()) {
        throw ToolArgumentError(tool_name, "'max_bytes' must be an integer");
      }
      const long long v = arguments.at("max_bytes").get<long long>();
      if (v < 1 || v > 20000) {
        throw ToolArgumentError(tool_name,
                               "'max_bytes' must be in range 1..20000");
      }
      max_bytes = static_cast<std::size_t>(v);
    }

    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
      throw ToolArgumentError(tool_name, "failed to open file: " + path);
    }

    std::string content;
    content.resize(max_bytes);
    file.read(content.data(), static_cast<std::streamsize>(max_bytes));
    const std::streamsize n = file.gcount();
    if (n < 0) {
      throw ToolArgumentError(tool_name, "failed while reading file: " + path);
    }
    content.resize(static_cast<std::size_t>(n));

    bool truncated = false;
    if (static_cast<std::size_t>(n) == max_bytes) {
      const int next = file.peek();
      truncated = (next != std::char_traits<char>::eof());
    }

    json out;
    out["path"] = path;
    out["max_bytes"] = max_bytes;
    out["truncated"] = truncated;
    out["content"] = content;
    return out.dump();
  }

  std::string get_name() const override { return "read_text_file"; }
};

class RunBashTool final : public Tool
{
public:
  common_chat_tool get_definition() const override
  {
    return { .name = "run_bash",
             .description =
               "Execute a bash command and return its stdout/stderr. "
               "Runs sandboxed via bubblewrap (bwrap) when available: "
               "isolated filesystem (rw only in project dir), no network, "
               "30s timeout. Falls back to bare subprocess if bwrap "
               "is not installed.",
             .params = { { .name = "command",
                           .type = "string",
                           .description =
                             "The bash command to run. Examples: "
                             "'ls -la', 'grep -r foo src/', "
                             "'cat file.txt', 'git diff --stat', "
                             "'cmake --build build'.",
                           .required = true } } };
  }

  std::string execute(const json& arguments) override
  {
    static constexpr const char* kBlocked[] = {
      "rm ", "sudo ", "chmod ", "chown ", "mkfs", "dd if=", ":(){",
    };
    const std::string tool_name = get_name();

    if (!arguments.contains("command") ||
        !arguments.at("command").is_string()) {
      throw ToolArgumentError(tool_name, "'command' must be a string");
    }

    const std::string cmd = arguments.at("command").get<std::string>();
    if (cmd.empty()) {
      throw ToolArgumentError(tool_name, "'command' must not be empty");
    }

    for (const auto& blocked : kBlocked) {
      if (cmd.find(blocked) != std::string::npos) {
        throw ToolArgumentError(tool_name,
                                "blocked dangerous command: " + cmd);
      }
    }

    const std::string wrapped = build_sandbox_cmd(cmd);

    std::array<char, 128> buf{};
    std::string output;
    output.reserve(4096);

    FILE* pipe = popen(wrapped.c_str(), "r");
    if (pipe == nullptr) {
      throw ToolArgumentError(tool_name, "failed to execute: " + cmd);
    }

    static constexpr size_t kMaxOutput = 10000;
    while (fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
      output += buf.data();
      if (output.size() > kMaxOutput) {
        output.resize(kMaxOutput);
        break;
      }
    }

    int rc = pclose(pipe);
    int exit_code = WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
    bool truncated = output.size() >= kMaxOutput;

    json out;
    out["command"] = cmd;
    out["sandbox"] = sandbox_available_;
    out["exit_code"] = exit_code;
    out["truncated"] = truncated;
    out["output"] = output;
    return out.dump();
  }

  std::string get_name() const override { return "run_bash"; }

private:
  static std::string build_sandbox_cmd(const std::string& cmd)
  {
    if (sandbox_available_) {
      return bwrap_prefix() + "timeout 30 bash -c '" +
             escape_sq(cmd) + "' 2>&1";
    }
    return "timeout 30 bash -c '" + escape_sq(cmd) + "' 2>&1";
  }

  static std::string bwrap_prefix()
  {
    const char* cwd = sandbox_root_.c_str();

    // clang-format off
    return std::string("bwrap ")
      + "--ro-bind /usr /usr "
      + "--ro-bind /lib /lib "
      + "--ro-bind /lib64 /lib64 "
      + "--ro-bind /bin /bin "
      + "--ro-bind /etc /etc "
      + "--ro-bind /opt /opt "
      + "--bind " + cwd + " " + cwd + " "
      + "--chdir " + cwd + " "
      + "--unshare-net "
      + "--unshare-pid "
      + "--die-with-parent "
      + "--dev /dev "
      + "--proc /proc "
      + "--tmpfs /tmp "
      + "--tmpfs /var "
      + "--tmpfs /run "
      + "--clearenv "
      + "--setenv HOME " + cwd + " "
      + "--setenv PATH /usr/bin:/bin:/usr/local/bin "
      + "--setenv USER agent "
      + "--setenv TERM xterm-256color "
      + "-- ";
    // clang-format on
  }

  static std::string escape_sq(const std::string& s)
  {
    std::string r;
    r.reserve(s.size() + 8);
    for (char c : s) {
      if (c == '\'') r += "'\\''";
      else r += c;
    }
    return r;
  }

  static bool detect_bwrap()
  {
    FILE* p = popen("bwrap --version 2>/dev/null", "r");
    if (p == nullptr) return false;
    char buf[64]{};
    bool ok = fgets(buf, sizeof(buf), p) != nullptr;
    pclose(p);
    return ok;
  }

  static std::string detect_cwd()
  {
    return std::filesystem::current_path().string();
  }

  static const bool sandbox_available_;
  static const std::string sandbox_root_;
};

const bool RunBashTool::sandbox_available_ = RunBashTool::detect_bwrap();
const std::string RunBashTool::sandbox_root_ = RunBashTool::detect_cwd();

struct BuiltinToolRegistrar
{
  BuiltinToolRegistrar()
  {
    (void)ToolRegistry::register_factory(
      "echo", []() { return std::make_unique<EchoTool>(); });
    (void)ToolRegistry::register_factory(
      "add", []() { return std::make_unique<AddTool>(); });
    (void)ToolRegistry::register_factory(
      "read_text_file", []() { return std::make_unique<ReadTextFileTool>(); });
    (void)ToolRegistry::register_factory(
      "run_bash", []() { return std::make_unique<RunBashTool>(); });
  }
};

const BuiltinToolRegistrar g_builtin_tool_registrar;

} // namespace

bool
ToolRegistry::register_factory(const std::string& name, Factory factory)
{
  if (name.empty() || !factory) {
    return false;
  }

  std::scoped_lock lock(registry_mutex());
  auto& map = registry_map();
  auto [it, inserted] = map.emplace(name, std::move(factory));
  (void)it;
  return inserted;
}

std::unique_ptr<Tool>
ToolRegistry::create(const std::string& name)
{
  std::scoped_lock lock(registry_mutex());
  auto& map = registry_map();
  auto it = map.find(name);
  if (it == map.end()) {
    return nullptr;
  }
  return it->second();
}

std::vector<std::string>
ToolRegistry::list_names()
{
  std::scoped_lock lock(registry_mutex());
  auto& map = registry_map();

  std::vector<std::string> names;
  names.reserve(map.size());
  for (const auto& kv : map) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::vector<common_chat_tool>
ToolRegistry::list_definitions()
{
  // Copy factories so we don't hold the lock while constructing tools.
  std::vector<Factory> factories;
  {
    std::scoped_lock lock(registry_mutex());
    auto& map = registry_map();
    factories.reserve(map.size());
    for (const auto& kv : map) {
      factories.push_back(kv.second);
    }
  }

  std::vector<common_chat_tool> defs;
  defs.reserve(factories.size());
  for (auto& f : factories) {
    auto tool = f();
    if (tool) {
      defs.push_back(tool->get_definition());
    }
  }
  return defs;
}

} // namespace zato
