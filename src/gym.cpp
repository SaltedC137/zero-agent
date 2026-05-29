#include "agent.hpp"
#include "chat.hpp"
#include "error.hpp"
#include "model.hpp"
#include "tool.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace {

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
  const std::string default_model_path =
    "model/Qwen2.5-Coder-3B-Instruct-Q8_0.gguf";
  std::string model_path = default_model_path;
  bool model_path_set = false;

  std::string system_prompt_path = "prompt/Qwen_artifacts_20250501.md";
  bool use_agent = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--system-prompt" && i + 1 < argc) {
      system_prompt_path = argv[++i];
      continue;
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
    config.n_ctx = 32768;
    config.n_batch = 2048;

    auto model = zato::Model::create(model_path, config);

    std::cout << "Zero-Agent gym ready. Model: " << model_path << std::endl;
    std::cout << "Type 'exit' to quit." << std::endl;

    std::vector<zato::common_chat_msg> messages;
    std::unique_ptr<zato::Agent> agent;

    if (use_agent) {
      std::vector<std::unique_ptr<zato::Tool>> tools;
      for (const std::string& name : { std::string("echo"),
                                      std::string("add"),
                                      std::string("read_text_file") }) {
        auto tool = zato::ToolRegistry::create(name);
        if (!tool) {
          throw zato::Error("Tool not registered: " + name);
        }
        tools.push_back(std::move(tool));
      }

      agent = std::make_unique<zato::Agent>(
          model,
          std::move(tools),
          std::vector<std::unique_ptr<zato::Callback>>{},
          load_system_prompt_file(system_prompt_path));
    } else {
      messages.push_back(
        zato::make_system_msg(load_system_prompt_file(system_prompt_path)));
    }

    std::string user_input;
    while (true) {
      std::cout << "You> ";
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

      if (use_agent) {
        std::cout << "AI> " << std::flush;
        agent->run_loop(messages, [](const std::string& delta) {
          std::cout << delta << std::flush;
        });
        std::cout << "\n";
      } else {
        std::vector<zato::common_chat_tool> tools;

        std::cout << "AI> " << std::flush;
        auto response =
            model->generate(messages, tools, [](const std::string &delta) {
              std::cout << delta << std::flush;
            });
        std::cout << "\n";

        messages.push_back(response);
      }
    }
  } catch (const zato::Error& e) {
    std::cerr << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
