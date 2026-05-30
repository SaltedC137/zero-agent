#include "window.hpp"
#include <iostream>

namespace zato {

WindowManager::WindowManager(std::shared_ptr<IModel> model,
                             int budget,
                             bool api_mode,
                             const std::string& system_prompt)
  : model_(std::move(model))
  , budget_(budget)
  , api_mode_(api_mode)
  , system_prompt_(system_prompt)
{
}

int
WindowManager::usage(const std::vector<common_chat_msg>& messages) const
{
  // API mode: rough char-based estimate (~3 chars per token for Chinese,
  // ~4 for English, use 3 as average)
  if (api_mode_) {
    size_t total = 0;
    for (const auto& m : messages) {
      total += m.content.size();
    }
    return static_cast<int>(total / 3);
  }
  // Local mode: accurate via tokenize
  return static_cast<int>(format_chatml(messages).size() /
                          3); // fallback — accurate version
                              // calls model->tokenize
}

bool
WindowManager::manage(std::vector<common_chat_msg>& messages)
{
  if (messages.empty()) {
    return false;
  }

  const int threshold = budget_ * 80 / 100; // trigger at 80%
  int used = usage(messages);
  if (used < threshold) {
    return false;
  }

  // API mode: summarize; local mode: discard
  if (api_mode_) {
    return compact(messages);
  }
  trim(messages);
  return true;
}

void
WindowManager::trim(std::vector<common_chat_msg>& messages)
{
  const size_t sys_end = (messages[0].role == MessageRole::SYSTEM) ? 1U : 0U;

  int total = usage(messages);
  const int limit = budget_ * 80 / 100;
  size_t keep_from = sys_end;
  const size_t min_keep = 4;

  while (keep_from + min_keep < messages.size() && total > limit) {
    total -= static_cast<int>(messages[keep_from].content.size() / 3);
    ++keep_from;
  }

  if (keep_from > sys_end) {
    messages.erase(messages.begin() + static_cast<long>(sys_end),
                   messages.begin() + static_cast<long>(keep_from));
  }
}

bool
WindowManager::compact(std::vector<common_chat_msg>& messages)
{
  const size_t sys_end = (messages[0].role == MessageRole::SYSTEM) ? 1U : 0U;
  if (messages.size() <= sys_end + 6) {
    return false;
  }

  // Find safe cut: most recent USER message (preserve tool_use/result pairs)
  size_t cut = messages.size();
  for (size_t i = messages.size() - 1; i > sys_end; --i) {
    if (messages[i].role == MessageRole::USER &&
        messages[i].tool_call_id.empty()) {
      cut = i;
      break;
    }
  }
  if (cut <= sys_end || cut >= messages.size()) {
    return false;
  }

  // Build summary prompt (limit input size)
  static constexpr size_t kMaxInput = 3000;
  std::string prompt =
    "Summarize this conversation in 2-3 sentences, keeping key facts "
    "(names, decisions, code context). Same language:\n\n";

  for (size_t i = sys_end; i < cut; ++i) {
    if (!messages[i].content.empty()) {
      std::string line = role_to_string(messages[i].role) + ": " +
                         messages[i].content.substr(0, 300) + "\n";
      if (prompt.size() + line.size() > kMaxInput) {
        break;
      }
      prompt += line;
    }
  }

  std::cerr << "\r  [compacting...]" << std::flush;

  std::vector<common_chat_msg> req;
  req.push_back(make_system_msg("Summarize. Output ONLY the summary."));
  req.push_back(make_user_msg(prompt));

  auto response = model_->generate(req, {}, nullptr);
  std::cerr << " done]\n" << std::flush;

  if (response.content.empty()) {
    return false;
  }

  // Replace old messages with summary (assistant role — Claude Code style)
  common_chat_msg summary;
  summary.role = MessageRole::ASSISTANT;
  summary.content = "[Context memory: " + response.content + "]";

  messages.erase(messages.begin() + static_cast<long>(sys_end),
                 messages.begin() + static_cast<long>(cut));
  messages.insert(messages.begin() + static_cast<long>(sys_end), summary);

  // Ensure system prompt is first
  if (messages.empty() || messages[0].content != system_prompt_) {
    messages.insert(messages.begin(), make_system_msg(system_prompt_));
  }

  return true;
}

} // namespace zato
