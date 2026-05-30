#include "memory.hpp"

namespace zato {

MemoryManager::MemoryManager(std::shared_ptr<IModel> model,
                             const std::string& system_prompt,
                             size_t max_chars)
  : model_(std::move(model))
  , system_prompt_(system_prompt)
  , max_chars_(max_chars)
{
}

void
MemoryManager::compact(std::vector<common_chat_msg>& messages)
{
  if (messages.empty()) return;

  size_t total = 0;
  for (const auto& m : messages) total += m.content.size();
  if (total < max_chars_) return;

  const size_t sys_end =
    (messages[0].role == MessageRole::SYSTEM) ? 1U : 0U;
  if (messages.size() <= sys_end + 6) return;

  // Build summary request from old messages
  std::string prompt =
    "Summarize the following conversation history in 2-3 sentences. "
    "Keep all key facts: names, preferences, decisions, code context. "
    "Write in the same language as the conversation.\n\n";

  for (size_t i = sys_end; i < messages.size() - 6; ++i) {
    if (!messages[i].content.empty()) {
      prompt += role_to_string(messages[i].role) + ": " +
                messages[i].content + "\n";
    }
  }

  // Call model to summarize
  std::vector<common_chat_msg> req;
  req.push_back(make_system_msg(
    "You are a conversation summarizer. Output ONLY the summary."));
  req.push_back(make_user_msg(prompt));

  auto response = model_->generate(req, {}, nullptr);

  if (response.content.empty()) return;

  // Replace old messages with summary
  messages.erase(messages.begin() + static_cast<long>(sys_end),
                 messages.begin() +
                   static_cast<long>(messages.size() - 6));
  common_chat_msg mem;
  mem.role = MessageRole::SYSTEM;
  mem.content = "[Conversation memory]: " + response.content;
  messages.insert(messages.begin() + static_cast<long>(sys_end), mem);

  // Ensure system prompt stays first
  if (messages[0].content != system_prompt_) {
    messages.insert(messages.begin(), make_system_msg(system_prompt_));
  }
}

} // namespace zato
