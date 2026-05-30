#include "agent.hpp"

#include "chat.hpp"
#include "error.hpp"
#include "imodel.hpp"
#include "tool.hpp"
#include "tool_result.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace zato {

using json = nlohmann::json;

Agent::Agent(std::shared_ptr<IModel> model,
             std::vector<std::unique_ptr<Tool>> tools,
             std::vector<std::unique_ptr<Callback>> callbacks,
             const std::string& instructions)
  : model(std::move(model))
  , tools(std::move(tools))
  , callbacks_(std::move(callbacks))
  , instructions(instructions)
{
}

void
Agent::ensure_system_message(std::vector<common_chat_msg>& messages)
{
  if (!instructions.empty()) {
    bool has_instructions = !messages.empty() &&
                            messages[0].role == MessageRole::SYSTEM &&
                            messages[0].content == instructions;

    if (!has_instructions) {
      common_chat_msg system_msg;
      system_msg.role = MessageRole::SYSTEM;
      system_msg.content = instructions;
      messages.insert(messages.begin(), system_msg);
    }
  }
}

std::vector<common_chat_tool>
Agent::get_tool_definitions() const
{
  std::vector<common_chat_tool> tool_definitions;
  tool_definitions.reserve(tools.size());
  for (const auto& tool : tools) {
    tool_definitions.push_back(tool->get_definition());
  }
  return tool_definitions;
}

std::string
Agent::run_loop(std::vector<common_chat_msg>& messages,
                const ResponseCallback& callback)
{

  ensure_system_message(messages);

  for (const auto& cb : callbacks_) {
    cb->before_agent_loop(messages);
  }

  std::vector<common_chat_tool> tool_definitions = get_tool_definitions();

  static constexpr int kMaxIterations = 25;
  static constexpr size_t kMaxToolOutput = 2000;
  std::vector<common_chat_tool_call> last_tool_calls;

  // Helper: extract commands from ```bash / ```sh blocks in model output.
  // When the 3B model forgets to use run_bash, we treat code blocks as
  // implicit tool calls so the user still gets real execution.
  auto find_bash_blocks = [](const std::string& text)
    -> std::vector<std::string> {
    std::vector<std::string> cmds;
    const std::string fence = "```";
    size_t pos = 0;
    while (true) {
      const size_t open = text.find(fence + "bash", pos);
      const size_t open_sh = text.find(fence + "sh", pos);
      size_t start = std::string::npos;
      if (open != std::string::npos && open_sh != std::string::npos) {
        start = std::min(open, open_sh);
      } else if (open != std::string::npos) {
        start = open;
      } else {
        start = open_sh;
}
      if (start == std::string::npos) { break;
}

      const size_t nl = text.find('\n', start);
      if (nl == std::string::npos) { break;
}
      const size_t close = text.find(fence, nl + 1);
      if (close == std::string::npos) break;

      std::string cmd =
        zato::trim_copy(text.substr(nl + 1, close - nl - 1));
      if (!cmd.empty()) cmds.push_back(std::move(cmd));
      pos = close + 3;
    }
    return cmds;
  };

  // Helper: find and execute a tool by name + args, returns result
  auto execute_tool = [&](const std::string& tname, const std::string& targs)
    -> ToolResult {
    ToolResult r("");
    try {
      for (const auto& cb : callbacks_) {
        cb->before_tool_execution(const_cast<std::string&>(tname),
                                  const_cast<std::string&>(targs));
      }
    } catch (const ToolExecutionSkipped& e) {
      json resp;
      resp["skipped"] = e.get_message();
      return ToolResult(resp.dump());
    }
    try {
      json args;
      try {
        args = json::parse(targs);
      } catch (const json::parse_error& e) {
        throw ToolArgumentError(tname, e.what());
      }
      auto it = std::find_if(
        tools.begin(), tools.end(),
        [&](const std::unique_ptr<Tool>& t) { return t->get_name() == tname; });
      if (it == tools.end()) throw ToolNotFoundError(tname);
      r = (*it)->execute(args);
    } catch (const std::exception& e) {
      r = ToolResult::from_exception(e);
    }
    for (const auto& cb : callbacks_) {
      cb->after_tool_execution(const_cast<std::string&>(tname), r);
    }
    return r;
  };

  auto tool_calls_eq = [](const std::vector<common_chat_tool_call>& a,
                          const std::vector<common_chat_tool_call>& b) -> bool {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i].tool_name != b[i].tool_name ||
          a[i].tool_args != b[i].tool_args) {
        return false;
      }
    }
    return true;
  };

  for (int iter = 0; iter < kMaxIterations; ++iter) {
    for (const auto& cb : callbacks_) {
      cb->before_llm_call(messages);
    }

    auto parsed_msg = model->generate(messages, tool_definitions, callback);

    for (const auto& cb : callbacks_) {
      cb->after_llm_call(parsed_msg);
    }

    messages.push_back(parsed_msg);

    // If model returned text without tool calls, check for implicit
    // ```bash blocks and treat them as run_bash calls automatically.
    if (parsed_msg.tool_calls.empty()) {
      const auto bash_cmds = find_bash_blocks(parsed_msg.content);
      if (!bash_cmds.empty()) {
        for (size_t ci = 0; ci < bash_cmds.size(); ++ci) {
          json args;
          args["command"] = bash_cmds[ci];
          const std::string call_id =
            "implicit_" + std::to_string(iter) + "_" + std::to_string(ci);
          ToolResult result = execute_tool("run_bash", args.dump());
          common_chat_msg tmsg;
          tmsg.role = MessageRole::TOOL;
          tmsg.tool_call_id = call_id;
          if (result.has_error()) {
            json ej;
            ej["error"] = result.error().message;
            tmsg.content = ej.dump();
          } else {
            std::string out = result.output();
            if (out.size() > kMaxToolOutput) {
              out.resize(kMaxToolOutput);
              out += "\n... [truncated]";
            }
            tmsg.content = std::move(out);
          }
          messages.push_back(tmsg);
        }
        continue; // loop back so model can respond to tool results
      }

      std::string response = parsed_msg.content;
      for (const auto& cb : callbacks_) {
        cb->after_agent_loop(messages, response);
      }
      return response;
    }

    // Detect repeated tool calls — model is stuck in a loop
    if (tool_calls_eq(parsed_msg.tool_calls, last_tool_calls)) {
      return "[agent stopped: repeated tool call detected]";
    }
    last_tool_calls = parsed_msg.tool_calls;

    for (const auto& tool_call : parsed_msg.tool_calls) {
      ToolResult result = execute_tool(tool_call.tool_name,
                                       tool_call.tool_args);
      common_chat_msg tool_msg;
      tool_msg.role = MessageRole::TOOL;
      tool_msg.tool_call_id = tool_call.tool_call_id;
      if (result.has_error()) {
        json error_json;
        error_json["error"] = result.error().message;
        tool_msg.content = error_json.dump();
      } else {
        std::string out = result.output();
        if (out.size() > kMaxToolOutput) {
          out.resize(kMaxToolOutput);
          out += "\n... [truncated]";
        }
        tool_msg.content = std::move(out);
      }
      messages.push_back(tool_msg);
    }
  }

  return "[agent stopped: max iterations reached]";
}

} // namespace zato
