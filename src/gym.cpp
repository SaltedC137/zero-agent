#include "agent.hpp"
#include "ansi.hpp"
#include "chat.hpp"
#include "context.hpp"
#include "error.hpp"
#include "model.hpp"
#include "session.hpp"
#include "tool.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

// Confirms bash commands before execution
class BashReviewCallback final : public zato::Callback
{
public:
  void before_tool_execution(std::string& tool_name,
                             std::string& arguments) override
  {
    if (tool_name != "run_bash") {
      return;
    }

    try {
      auto args = nlohmann::json::parse(arguments);
      std::string cmd = args.value("command", arguments);
      std::cout << zato::ansi::kYellow << "\n  $ " << cmd << zato::ansi::kReset
                << "\n"
                << "  Execute? [y/N] " << std::flush;

      std::string answer;
      if (!std::getline(std::cin, answer) ||
          (answer != "y" && answer != "Y" && answer != "yes")) {
        throw zato::ToolExecutionSkipped("user rejected: " + cmd);
      }
    } catch (const zato::ToolExecutionSkipped&) {
      throw;
    } catch (...) {
      // Can't parse — let the tool handle it
    }
  }
};

std::string
load_system_prompt_file(const std::string& path)
{
  std::ifstream file(path);
  if (!file.is_open()) {
    throw zato::Error("Failed to open system prompt file: " + path);
  }

  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  return zato::trim_copy(content);
}

} // namespace

int
main(int argc, char** argv)
{
  // Resolve executable directory early (needed by --list-sessions)
  std::string exe_path(256, '\0');
  const ssize_t l = readlink("/proc/self/exe", exe_path.data(), 255);
  const auto exe_dir =
    (l > 0) ? std::filesystem::path(exe_path.substr(0, static_cast<size_t>(l)))
                .parent_path()
            : std::filesystem::current_path();
  const auto sessions_dir = exe_dir / ".zato" / "sessions";

  const std::string default_model_path =
    "model/Qwen2.5-Coder-3B-Instruct-Q8_0.gguf";
  std::string model_path = default_model_path;
  bool model_path_set = false;

  std::string system_prompt_path = "prompt/Qwen_artifacts_20250501.md";
  std::string session_name = "default";
  bool use_agent = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--system-prompt" && i + 1 < argc) {
      system_prompt_path = argv[++i];
      continue;
    }

    if (arg == "--session" && i + 1 < argc) {
      session_name = argv[++i];
      continue;
    }

    if (arg == "--list-sessions") {
      if (std::filesystem::exists(sessions_dir)) {
        for (const auto& entry :
             std::filesystem::directory_iterator(sessions_dir)) {
          if (entry.is_directory()) {
            std::cout << "  " << entry.path().filename().string() << "\n";
          }
        }
      } else {
        std::cout << "  (no saved sessions)\n";
      }
      return 0;
    }

    if (arg == "--agent") {
      use_agent = true;
      continue;
    }

    // Unknown flags are ignored for now.
    if (arg.rfind("--", 0) == 0) {
      continue;
    }

    // First positional argument is treated as the model path.
    if (!model_path_set) {
      model_path = arg;
      model_path_set = true;
      continue;
    }
  }

  try {
    zato::ModelConfig config;
    config.temp = 0.0f;
    config.top_k = 1;
    config.max_p = 1.0f;
    config.min_p = 0.0f;
    config.n_ctx = 16384;
    config.n_batch = 2048;

    auto model = zato::Model::create(model_path, config);

    zato::SessionManager session(model, exe_dir, session_name);
    zato::ContextManager ctx(model, config.n_ctx);

    // Restore previous session if available
    auto messages = session.load();
    const std::string system_prompt =
      load_system_prompt_file(system_prompt_path);

    if (!messages.empty()) {
      std::cout << zato::ansi::kDim << "[" << session_name
                << ": restored " << messages.size() << " messages]"
                << zato::ansi::kReset << "\n";
      // Replace system message with current prompt file content
      if (messages[0].role == zato::MessageRole::SYSTEM) {
        messages[0].content = system_prompt;
      }
    }

    std::cout << zato::ansi::kBold << "Zero-Agent gym ready."
              << zato::ansi::kReset << " Model: " << zato::ansi::kCyan
              << model_path << zato::ansi::kReset << "\n"
              << "Type " << zato::ansi::kYellow << "'exit'"
              << zato::ansi::kReset << " to quit.\n";

    std::unique_ptr<zato::Agent> agent;

    if (use_agent) {
      std::vector<std::unique_ptr<zato::Tool>> tools;
      for (const std::string& name : { std::string("echo"),
                                       std::string("add"),
                                       std::string("read_text_file"),
                                       std::string("run_bash") }) {
        auto tool = zato::ToolRegistry::create(name);
        if (!tool) {
          throw zato::Error("Tool not registered: " + name);
        }
        tools.push_back(std::move(tool));
      }

      std::vector<std::unique_ptr<zato::Callback>> callbacks;
      callbacks.push_back(std::make_unique<BashReviewCallback>());

      agent = std::make_unique<zato::Agent>(
        model, std::move(tools), std::move(callbacks), system_prompt);
    } else if (messages.empty()) {
      messages.push_back(zato::make_system_msg(system_prompt));
    }

    std::string user_input;
    while (true) {
      std::cout << zato::ansi::kGreen << "You> " << zato::ansi::kReset;
      if (!std::getline(std::cin, user_input)) {
        break;
      }
      if (user_input == "exit") {
        break;
      }
      if (user_input.empty()) {
        continue;
      }

      messages.push_back(zato::make_user_msg(user_input));

      ctx.trim(messages);

      if (use_agent) {
        std::cout << zato::ansi::kMagenta << "AI> " << zato::ansi::kReset
                  << std::flush;
        agent->run_loop(messages, [](const std::string& delta) {
          std::cout << delta << std::flush;
        });
        std::cout << "\n";
      } else {
        std::vector<zato::common_chat_tool> tools;

        std::cout << zato::ansi::kMagenta << "AI> " << zato::ansi::kReset
                  << std::flush;
        auto response =
          model->generate(messages, tools, [](const std::string& delta) {
            std::cout << delta << std::flush;
          });
        std::cout << "\n";

        messages.push_back(response);
      }

      session.save(messages);
    }
  } catch (const zato::Error& e) {
    std::cerr << zato::ansi::kRed << e.what() << zato::ansi::kReset << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << zato::ansi::kRed << "Error: " << e.what() << zato::ansi::kReset
              << "\n";
    return 1;
  }

  return 0;
}
