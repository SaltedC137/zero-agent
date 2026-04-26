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
  std::string response_content;
  std::string tool_name;
  std::string tool_call_id;

  bool empty() const {
    return role.empty() && content.empty() && content_parts.empty() &&
           tool_calls.empty() && response_content.empty() &&
           tool_name.empty() && tool_call_id.empty();
  }

  bool operator==(const common_chat_msg &other) const {
    return role == other.role && content == other.content &&
           content_parts == other.content_parts &&
           tool_calls == other.tool_calls &&
           response_content == other.response_content &&
           tool_name == other.tool_name && tool_call_id == other.tool_call_id;
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
    if (!response_content.empty()) {
      j["response_content"] = response_content;
    }
    if (!tool_name.empty()) {
      j["tool_name"] = tool_name;
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
    if (j.contains("response_content")) {
      msg.response_content = j.at("response_content").get<std::string>();
    }
    if (j.contains("tool_name")) {
      msg.tool_name = j.at("tool_name").get<std::string>();
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
};
