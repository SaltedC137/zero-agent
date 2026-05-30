#include "agent.hpp"
#include "ansi.hpp"
#include "api.hpp"
#include "chat.hpp"
#include "context.hpp"
#include "error.hpp"
#include "memory.hpp"
#include "model.hpp"
#include "session.hpp"
#include "tool.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <locale>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

class CompactCallback final : public zato::Callback
{
public:
  explicit CompactCallback(zato::MemoryManager& m)
    : mem_(m)
  {
  }
  void before_llm_call(std::vector<zato::common_chat_msg>& msgs) override
  {
    mem_.compact(msgs);
  }

private:
  zato::MemoryManager& mem_;
};

class BashReviewCallback final : public zato::Callback
{
public:
  void before_tool_execution(std::string& tool_name,
                             std::string& arguments) override
  {
    if (tool_name != "run_bash") {
      return;
    }
    std::string cmd;
    try {
      auto args = nlohmann::json::parse(arguments);
      cmd = args.value("command", arguments);
    } catch (...) {
      cmd = arguments;
    }
    std::cout << zato::ansi::kYellow << "\n  $ " << cmd << zato::ansi::kReset
              << "\n  Execute? [y/N] " << std::flush;
    std::string answer;
    if (!std::getline(std::cin, answer) ||
        (answer != "y" && answer != "Y" && answer != "yes")) {
      throw zato::ToolExecutionSkipped("user rejected: " + cmd);
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
  std::locale::global(std::locale("")); // UTF-8 aware terminal I/O
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

  int gpu_layers = 999;
  std::string system_prompt_path = "prompt/Qwen_artifacts_20250501.md";
  std::string session_name = "default";
  bool use_agent = false;

  // API mode via Anthropic-compatible env vars
  const char* env_url = std::getenv("ANTHROPIC_BASE_URL");
  bool use_api = (env_url != nullptr && env_url[0] != '\0');

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
    if (arg == "--gpu-layers" && i + 1 < argc) {
      gpu_layers = std::stoi(argv[++i]);
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
    if (arg == "--api") {
      use_api = true;
      continue;
    }
    if (arg.rfind("--", 0) == 0) {
      continue;
    }
    if (!model_path_set) {
      model_path = arg;
      model_path_set = true;
      continue;
    }
  }

  try {
    std::shared_ptr<zato::IModel> model;
    std::unique_ptr<zato::SessionManager> session;
    std::unique_ptr<zato::ContextManager> ctx;
    std::unique_ptr<zato::MemoryManager> mem;

    const std::string system_prompt =
      load_system_prompt_file(system_prompt_path);

    if (use_api) {
      zato::ApiModel::Config api_cfg;
      api_cfg.base_url = env_url;
      api_cfg.api_key = std::string(std::getenv("ANTHROPIC_AUTH_TOKEN") ?: "");
      api_cfg.model =
        std::string(std::getenv("ANTHROPIC_MODEL") ?: "claude-sonnet-4-6");
      model = std::make_shared<zato::ApiModel>(std::move(api_cfg));
      mem = std::make_unique<zato::MemoryManager>(model, system_prompt, 48000);

      std::cout << zato::ansi::kBold << "Zero-Agent API mode."
                << zato::ansi::kReset << " Model: " << zato::ansi::kCyan
                << api_cfg.model << zato::ansi::kReset << "\n";
    } else {
      zato::ModelConfig config;
      config.temp = 0.0f;
      config.top_k = 1;
      config.max_p = 1.0f;
      config.min_p = 0.0f;
      config.n_ctx = 16384;
      config.n_batch = 2048;

      int n_phys = static_cast<int>(std::thread::hardware_concurrency());
      if (n_phys > 8) {
        n_phys /= 2;
      }
      config.n_threads = n_phys;
      config.n_threads_batch = n_phys;
      config.n_gpu_layers = gpu_layers;

      auto local = zato::Model::create(model_path, config);
      model = local;

      session =
        std::make_unique<zato::SessionManager>(local, exe_dir, session_name);
      ctx = std::make_unique<zato::ContextManager>(local, config.n_ctx);

      std::cout << zato::ansi::kBold << "Zero-Agent gym ready."
                << zato::ansi::kReset << " Model: " << zato::ansi::kCyan
                << model_path << zato::ansi::kReset << "\n"
                << "Type " << zato::ansi::kYellow << "'exit'"
                << zato::ansi::kReset << " to quit.\n";
    }

    std::vector<zato::common_chat_msg> messages;

    if (session) {
      messages = session->load();
      if (!messages.empty()) {
        std::cout << zato::ansi::kDim << "[" << session_name << ": restored "
                  << messages.size() << " messages]" << zato::ansi::kReset
                  << "\n";
        if (messages[0].role == zato::MessageRole::SYSTEM) {
          messages[0].content = system_prompt;
        }
      }
    }

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
      if (use_api) {
        callbacks.push_back(std::make_unique<CompactCallback>(*mem));
      }
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
      if (ctx) {
        ctx->trim(messages);
      } else if (use_api) {
        mem->compact(messages);
      }

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

      if (session) {
        session->save(messages);
      }
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
