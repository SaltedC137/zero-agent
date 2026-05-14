#include "chat.hpp"
#include "error.hpp"
#include "model.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace zota {

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
    throw zota::Error("Failed to open system prompt file: " + path);
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

} // namespace zota

int main(int argc, char **argv) {
  const std::string model_path =
      argc > 1 ? argv[1] : "model/Qwen2.5-Coder-3B-Instruct-Q8_0.gguf";
  std::string system_prompt_path;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--system-prompt" && i + 1 < argc) {
      system_prompt_path = argv[++i];
    }
  }

  try {
    zota::ModelConfig config;
    config.temp = 0.0f;
    config.top_k = 1;
    config.max_p = 1.0f;
    config.min_p = 0.0f;
    config.n_ctx = 32768;
    config.n_batch = 2048;

    auto model = zota::Model::create(model_path, config);

    std::cout << "Zero-Agent gym ready. Model: " << model_path << std::endl;
    std::cout << "Type 'exit' to quit." << std::endl;

    std::vector<zota::common_chat_msg> messages;
    const std::string default_system_prompt =
        R"(You are Qwen, a professional coding assistant
created by Alibaba Cloud. You are proficient in many programming languages, frameworks, and
tools. You specialize in code generation, debugging, optimization, and explanation. Your
answers are accurate, concise, and focus on correctness, readability, and best practices. When
responding, you first analyze the problem, then provide clear reasoning and code examples. If
you are unsure or need more information, you will ask for clarification.)";

    if (!system_prompt_path.empty()) {
      messages.push_back(zota::make_system_msg(
          zota::load_system_prompt_file(system_prompt_path)));
    } else {
      messages.push_back(zota::make_system_msg(default_system_prompt));
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

      messages.push_back(zota::make_user_msg(user_input));
      std::vector<zota::common_chat_tool> tools;

      std::cout << "AI> " << std::flush;
      auto response =
          model->generate(messages, tools, [](const std::string &delta) {
            std::cout << delta << std::flush;
          });
      std::cout << std::endl;

      messages.push_back(response);
    }
  } catch (const zota::Error &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
