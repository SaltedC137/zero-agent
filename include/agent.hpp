#pragma once

#include "callback.hpp"
#include "chat.hpp"
#include "model.hpp"
#include "tool.hpp"

#include <llama.h>
#include <memory>
#include <string>
#include <vector>

namespace zato {

class Agent {
private:
  std::vector<std::unique_ptr<Callback>> callbacks_;
  std::string instructions;
  std::shared_ptr<Model> model;
  std::vector<std::unique_ptr<Tool>> tools;

  void ensure_system_message(std::vector<common_chat_msg> &messages);

public:
  Agent(std::shared_ptr<Model> model, std::vector<std::unique_ptr<Tool>> tools,
        std::vector<std::unique_ptr<Callback>> callbacks = {},
        const std::string &instructions = "");

  std::string run_loop(std::vector<common_chat_msg> &messages,
                       const ResponseCallback &callback = nullptr);

  [[nodiscard]] std::vector<common_chat_tool> get_tool_definitions() const;

  [[nodiscard]] const std::string &get_instructions() const {
    return instructions;
  }

  bool load_or_create_cache(const std::string &cache_path);

private:
  std::vector<llama_token> build_prompt_tokens();
};
} // namespace zato
// namespace zato
