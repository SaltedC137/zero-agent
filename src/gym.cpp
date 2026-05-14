#include "chat.hpp"
#include "error.hpp"
#include "model.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace zato {

// Trim leading and trailing whitespace from a string.
std::string trim(const std::string &text) {
  const auto begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }

  const auto end = text.find_last_not_of(" \t\r\n");
  return text.substr(begin, end - begin + 1);
}

std::string load_system_prompt_file(const std::string &path) {
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
}

} // namespace zato

int main(int argc, char **argv) {
  const std::string model_path =
      argc > 1 ? argv[1] : "model/Qwen2.5-Coder-3B-Instruct-Q8_0.gguf";
  std::string system_prompt_path = "prompt/Qwen_artifacts_20250501.md";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--system-prompt" && i + 1 < argc) {
      system_prompt_path = argv[++i];
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
    messages.push_back(zato::make_system_msg(
        zato::load_system_prompt_file(system_prompt_path)));

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
      std::vector<zato::common_chat_tool> tools;

      std::cout << "AI> " << std::flush;
      auto response =
          model->generate(messages, tools, [](const std::string &delta) {
            std::cout << delta << std::flush;
          });
      std::cout << std::endl;

      messages.push_back(response);
    }
  } catch (const zato::Error &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
