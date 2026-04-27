#pragma once

#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

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
};

struct common_chat_msg {
  std::string role;
  std::string content;
  std::vector<common_chat_msg_content_part> content_parts;
  std::vector<common_chat_tool_call> tool_calls;
  std::string reasoning_content;
  std::string tool_call_id;

  bool empty() const {
    return role.empty() && content.empty() && content_parts.empty() &&
           tool_calls.empty() && reasoning_content.empty() &&
           tool_call_id.empty();
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
    j["role"] = role;
    j["content"] = content;
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
    msg.role = j.at("role").get<std::string>();
    msg.content = j.at("content").get<std::string>();
    if (j.contains("content_parts")) {
      for (const auto &part : j.at("content_parts")) {
        msg.content_parts.push_back(
            common_chat_msg_content_part{part.at("type").get<std::string>(),
                                         part.at("text").get<std::string>()});
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

inline common_chat_msg make_user_msg(const std::string &content) {
  return {.role = "user", .content = content};
}

inline common_chat_msg make_assistant_msg(const std::string &content) {
  return {.role = "assistant", .content = content};
}

inline common_chat_msg make_system_msg(const std::string &content) {
  return {.role = "system", .content = content};
}

inline common_chat_msg make_tool_msg(const std::string &tool_call_id,
                                     const std::string &result) {
  return {.role = "tool",
          .content = result,
          .tool_call_id = tool_call_id};
}
