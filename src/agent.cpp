#include "agent.hpp"
#include "chat.hpp"

namespace zato {

using json = nlohmann::json;

Agent::Agent(std::shared_ptr<Model> model,
             std::vector<std::unique_ptr<Tool>> tools,
             std::vector<std::unique_ptr<Callback>> callbacks,
             const std::string &instructions)
    : model(std::move(model)), tools(std::move(tools)),
      callbacks_(std::move(callbacks)), instructions(instructions) {}

void Agent::ensure_system_message(std::vector<common_chat_msg> &messages) {
  if (!instructions.empty()) {
    bool has_instructions = !messages.empty() && messages[0].role == "system" &&
                            messages[0].content == instructions;

    if (!has_instructions) {
      common_chat_msg system_msg;
      system_msg.role = "system";
      system_msg.content = instructions;
      messages.insert(messages.begin(), system_msg);
    }
  }
}

} // namespace zato
