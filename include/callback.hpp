#pragma once

#include "chat.hpp"
#include <llama.h>
#include <string>

namespace zato {

class Callback
{
public:
  virtual ~Callback() = default;

  virtual void before_agent_loop(std::vector<zato::common_chat_msg>& messages)
  {
  }

  virtual void after_agent_loop(std::vector<zato::common_chat_msg>& messages,
                                std::string& response)
  {
  }

  // Callbacks for LLM calls. These can be used to modify the messages before
  // they

  virtual void before_llm_call(std::vector<common_chat_msg>& messages) {}

  virtual void after_llm_call(common_chat_msg& parsed_msg) {}

  virtual void before_tool_execution(std::string& tool_name,
                                     std::string& arguments)
  {
  }

  // Callbacks for tool execution. These can be used to modify the tool name and

  virtual void after_tool_execution(std::string& tool_name, std::string& result)
  {
    (void)tool_name;
    (void)result;
  }
};

} // namespace zato
