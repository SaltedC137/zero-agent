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
      tools.push_back(std::make_unique<EchoTool>());
      tools.push_back(std::make_unique<AddTool>());

      agent = std::make_unique<zato::Agent>(
          model,
          std::move(tools),
          std::vector<std::unique_ptr<zato::Callback>>{},
          zato::load_system_prompt_file(system_prompt_path));
    } else {
      messages.push_back(zato::make_system_msg(
          zato::load_system_prompt_file(system_prompt_path)));
    }

    std::string user_input;
    while (true) {
      std::cout << "\nYou> ";
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
        const std::string response = agent->run_loop(messages, nullptr);
        std::cout << response << std::endl;
      } else {
        std::vector<zato::common_chat_tool> tools;

        std::cout << "AI> " << std::flush;
        auto response =
            model->generate(messages, tools, [](const std::string &delta) {
              std::cout << delta << std::flush;
            });
        std::cout << std::endl;

        messages.push_back(response);
      }
    }
  } catch (const zato::Error& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
