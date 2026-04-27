/**
 * @file chat.hpp
 * @author Aska Lyn (saltedc137@gmail)
 * @brief structs and functions for chat management, including message
 * representation, tool calls, and message diffs.
 * @version 0.1
 * @date 2026-04-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once
#include <functional>
#include <llama.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace zota {

enum class MessageRole { USER, ASSISTANT, SYSTEM, TOOL };

enum common_chat_format {
  CONTENT_ONLY,
  WITH_TOOLS,
  WITH_REASONING,
};

inline MessageRole role_from_string(const std::string &s) {
  if (s == "user")
    return MessageRole::USER;
  if (s == "assistant")
    return MessageRole::ASSISTANT;
  if (s == "system")
    return MessageRole::SYSTEM;
  if (s == "tool")
    return MessageRole::TOOL;
  return MessageRole::USER;
}

inline std::string role_to_string(MessageRole role) {
  switch (role) {
  case MessageRole::USER:
    return "user";
  case MessageRole::ASSISTANT:
    return "assistant";
  case MessageRole::SYSTEM:
    return "system";
  case MessageRole::TOOL:
    return "tool";
  default:
    return "user";
  }
}

// This file defines the common_chat_msg struct, which is used to represent a
// message in a chat conversation.

struct common_chat_tool_call {
  std::string tool_name;
  std::string tool_args;
  std::string tool_call_id;

  bool operator==(const common_chat_tool_call &other) const {
    return tool_name == other.tool_name && tool_args == other.tool_args &&
           tool_call_id == other.tool_call_id;
  }

  json to_json() const {
    return json{{"tool_name", tool_name},
                {"tool_args", tool_args},
                {"tool_call_id", tool_call_id}};
  }

  static common_chat_tool_call from_json(const json &j) {
    return common_chat_tool_call{j.at("tool_name").get<std::string>(),
                                 j.at("tool_args").get<std::string>(),
                                 j.at("tool_call_id").get<std::string>()};
  }
};

struct common_chat_msg_content_part {
  std::string type;
  std::string text;

  bool operator==(const common_chat_msg_content_part &other) const {
    return type == other.type && text == other.text;
  }

  json to_json() const { return json{{"type", type}, {"text", text}}; }

  static common_chat_msg_content_part from_json(const json &j) {
    return {j.at("type").get<std::string>(), j.at("text").get<std::string>()};
  }
};

/**
 * @brief The common_chat_msg struct represents a message in a chat
 * conversation. It can contain text, tool calls, and reasoning content.
 *
 */

struct common_chat_msg {
  MessageRole role = MessageRole::USER;
  std::string content;
  std::vector<common_chat_msg_content_part> content_parts;
  std::vector<common_chat_tool_call> tool_calls;
  std::string reasoning_content;
  std::string tool_call_id;

  bool empty() const {
    return content.empty() && content_parts.empty() && tool_calls.empty() &&
           reasoning_content.empty() && tool_call_id.empty();
  }

  bool operator==(const common_chat_msg &other) const {
    return role == other.role && content == other.content &&
           content_parts == other.content_parts &&
           tool_calls == other.tool_calls &&
           reasoning_content == other.reasoning_content &&
           tool_call_id == other.tool_call_id;
  }

  bool operator!=(const common_chat_msg &other) const {
    return !(*this == other);
  }

  json to_json_oaicompat() const {
    json j;
    j["role"] = role_to_string(role);
    if (!content.empty()) {
      j["content"] = content;
    }
    if (!content_parts.empty()) {
      j["content_parts"] = json::array();
      for (const auto &part : content_parts) {
        j["content_parts"].push_back(part.to_json());
      }
    }
    if (!tool_calls.empty()) {
      j["tool_calls"] = json::array();
      for (const auto &call : tool_calls) {
        j["tool_calls"].push_back(call.to_json());
      }
    }
    if (!reasoning_content.empty()) {
      j["reasoning_content"] = reasoning_content;
    }
    if (!tool_call_id.empty()) {
      j["tool_call_id"] = tool_call_id;
    }
    return j;
  }

  static common_chat_msg from_json_oaicompat(const json &j) {
    common_chat_msg msg;
    msg.role = role_from_string(j.at("role").get<std::string>());
    if (j.contains("content")) {
      msg.content = j["content"].get<std::string>();
    }
    if (j.contains("content_parts")) {
      for (const auto &part : j.at("content_parts")) {
        msg.content_parts.push_back(
            common_chat_msg_content_part::from_json(part));
      }
    }
    if (j.contains("tool_calls")) {
      for (const auto &call : j.at("tool_calls")) {
        msg.tool_calls.push_back(common_chat_tool_call::from_json(call));
      }
    }
    if (j.contains("reasoning_content")) {
      msg.reasoning_content = j.at("reasoning_content").get<std::string>();
    }
    if (j.contains("tool_call_id")) {
      msg.tool_call_id = j.at("tool_call_id").get<std::string>();
    }
    return msg;
  }
};

// tool definitions are used to represent the tools that can be called in a chat

struct common_chat_tool_param {
  std::string name;
  std::string type;
  std::string description;
  bool required = false;
};

struct common_chat_tool {
  std::string name;
  std::string description;
  std::vector<common_chat_tool_param> params;

  json to_json_schema() const {
    json props = json::object();
    json required = json::array();

    for (const auto &param : params) {
      props[param.name] = {{"type", param.type},
                           {"description", param.description}};
      if (param.required) {
        required.push_back(param.name);
      }
    }

    return json{
        {"name", name},
        {"description", description},
        {"parameters",
         {{"type", "object"}, {"properties", props}, {"required", required}}}};
  }
};

// message manager is used to manage the messages in a chat conversation

struct common_chat_msg_diff {
  std::string content_delta;
  std::string reasoning_content_delta;
  size_t tool_call_index = static_cast<size_t>(-1);
  common_chat_tool_call tool_call;
};

// chat_template functions

struct common_chat_templates {
  std::string name;
  std::function<std::string(const std::vector<common_chat_msg> &)> format_func;

  std::string apply(const std::vector<common_chat_msg> &messages) const {
    return format_func(messages);
  }

  bool is_chatml() const { return name.find("chatml") != std::string::npos; }

  bool is_alpaca() const { return name.find("alpaca") != std::string::npos; }
};

// common chat templates are used to define the templates for formatting the
// chat

inline common_chat_msg make_user_msg(const std::string &content) {
  return {.role = MessageRole::USER, .content = content};
}

inline common_chat_msg make_assistant_msg(const std::string &content) {
  return {.role = MessageRole::ASSISTANT, .content = content};
}

inline common_chat_msg make_system_msg(const std::string &content) {
  return {.role = MessageRole::SYSTEM, .content = content};
}

inline common_chat_msg make_tool_msg(const std::string &tool_call_id,
                                     const std::string &result) {
  return {.role = MessageRole::TOOL,
          .content = result,
          .tool_call_id = tool_call_id};
}

// formatting functions for different chat templates

inline std::string format_chatml(const std::vector<common_chat_msg> &messages) {
  std::string prompt;
  for (const auto &msg : messages) {
    if (!msg.content.empty()) {
      prompt += "<|im_start|>" + role_to_string(msg.role) + "\n";
      prompt += msg.content + "\n";
      prompt += "<|im_end|>\n";
    }
  }
  prompt += "<|im_start|>" + role_to_string(MessageRole::ASSISTANT) + "\n";
  return prompt;
}

inline std::string format_alpaca(const std::vector<common_chat_msg> &messages) {
  std::string prompt = "Below is an instruction that describes a task.\n\n";
  for (const auto &msg : messages) {
    if (msg.role == MessageRole::USER) {
      prompt += "### Instruction:\n" + msg.content + "\n\n";
    } else if (msg.role == MessageRole::ASSISTANT) {
      prompt += "### Response:\n" + msg.content + "\n\n";
    }
  }
  prompt += "### Response:\n";
  return prompt;
}

inline std::string format_simple(const std::vector<common_chat_msg> &messages) {
  std::string prompt;
  for (const auto &msg : messages) {
    if (!msg.content.empty()) {
      prompt += role_to_string(msg.role) + ": " + msg.content + "\n";
    }
  }
  prompt += role_to_string(MessageRole::ASSISTANT) + ": ";
  return prompt;
}

inline std::string
format_by_model_name(const std::vector<common_chat_msg> &messages,
                     const std::string &model_path) {
  if (model_path.find("llama") != std::string::npos) {
    return format_alpaca(messages);
  } else if (model_path.find("mistral") != std::string::npos) {
    return format_chatml(messages);
  } else if (model_path.find("neural") != std::string::npos) {
    return format_chatml(messages);
  }
  return format_chatml(messages);
}

inline std::string
detect_and_format(const std::vector<common_chat_msg> &messages,
                  const llama_model *model) {
  if (model == nullptr) {
    return format_chatml(messages);
  }

  int buf_size =
      llama_model_meta_val_str(model, "tokenizer.chat_template", nullptr, 0);
  if (buf_size <= 0) {
    return format_chatml(messages);
  }

  std::string chat_template(buf_size, '\0');
  llama_model_meta_val_str(model, "tokenizer.chat_template", &chat_template[0],
                           buf_size);

  if (chat_template.find("chatml") != std::string::npos) {
    return format_chatml(messages);
  } else if (chat_template.find("alpaca") != std::string::npos) {
    return format_alpaca(messages);
  }

  return format_simple(messages);
}

inline bool should_process_tool_calls(const common_chat_msg &msg) {
  return msg.role == MessageRole::ASSISTANT && !msg.tool_calls.empty();
}

inline bool is_response_message(const common_chat_msg &msg) {
  return msg.role == MessageRole::ASSISTANT || msg.role == MessageRole::TOOL;
}

inline std::vector<common_chat_tool_call>
extract_tool_calls(const std::vector<common_chat_msg> &messages) {
  for (const auto &msg : messages) {
    if (!msg.tool_calls.empty()) {
      return msg.tool_calls;
    }
  }
  return {};
}

} // namespace zota