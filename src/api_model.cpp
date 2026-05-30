#include "api_model.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <stdexcept>

namespace zato {

using json = nlohmann::json;

ApiModel::ApiModel(Config cfg)
  : cfg_(std::move(cfg))
{
  is_anthropic_ = cfg_.base_url.find("/anthropic") != std::string::npos ||
                  cfg_.base_url.find("/v1/messages") != std::string::npos;
}

common_chat_msg
ApiModel::generate(const std::vector<common_chat_msg>& messages,
                   const std::vector<common_chat_tool>& tools,
                   ResponseCallback callback)
{
  if (is_anthropic_) {
    return generate_anthropic(messages, tools, callback);
  }
  return generate_openai(messages, tools, callback);
}

common_chat_msg
ApiModel::generate_openai(const std::vector<common_chat_msg>& msgs,
                          const std::vector<common_chat_tool>& tools,
                          ResponseCallback callback)
{
  json body;
  body["model"] = cfg_.model;
  body["stream"] = true;
  body["temperature"] = 0.0;
  body["max_tokens"] = 4096;
  body["messages"] = json::array();
  for (const auto& m : msgs) {
    body["messages"].push_back(m.to_json_oaicompat());
  }
  if (!tools.empty()) {
    body["tools"] = json::array();
    for (const auto& t : tools) {
      json tj;
      tj["type"] = "function";
      tj["function"] = t.to_json_schema();
      body["tools"].push_back(tj);
    }
  }

  auto [host, path] = split_url(cfg_.base_url, "/chat/completions");
  httplib::Client cli(host);
  cli.set_read_timeout(120);

  httplib::Headers hdrs = { { "Content-Type", "application/json" } };
  if (!cfg_.api_key.empty()) {
    hdrs.emplace("Authorization", "Bearer " + cfg_.api_key);
  }

  std::string full;
  json acc = json::array();

  auto res = cli.Post(path,
                      hdrs,
                      body.dump(),
                      "application/json",
                      [&](const char* d, size_t n) -> bool {
                        return parse_sse(std::string(d, n), full, acc, callback);
                      });

  if (!res) {
    return make_assistant_msg(
      "[API error " + std::to_string(-1) + "]");
  }

  return build_msg(full, acc);
}

common_chat_msg
ApiModel::generate_anthropic(const std::vector<common_chat_msg>& msgs,
                             const std::vector<common_chat_tool>& tools,
                             ResponseCallback callback)
{
  json body;
  body["model"] = cfg_.model;
  body["stream"] = true;
  body["temperature"] = 0.0;
  body["max_tokens"] = 4096;

  body["messages"] = json::array();
  for (size_t i = 0; i < msgs.size(); ++i) {
    const auto& m = msgs[i];
    if (m.role == MessageRole::SYSTEM) {
      body["system"] = m.content;
      continue;
    }
    json jm;
    jm["role"] = role_to_string(m.role);
    if (!m.tool_calls.empty()) {
      jm["content"] = json::array();
      if (!m.content.empty()) {
        jm["content"].push_back({{"type", "text"}, {"text", m.content}});
      }
      for (const auto& tc : m.tool_calls) {
        json tcu;
        tcu["type"] = "tool_use";
        tcu["id"] = tc.tool_call_id;
        tcu["name"] = tc.tool_name;
        try {
          tcu["input"] = json::parse(tc.tool_args);
        } catch (...) {
          tcu["input"] = json::object();
        }
        jm["content"].push_back(tcu);
      }
    } else if (m.role == MessageRole::TOOL) {
      // Merge consecutive TOOL messages into a single user message
      jm["role"] = "user";
      jm["content"] = json::array();
      while (i < msgs.size() && msgs[i].role == MessageRole::TOOL) {
        json tr;
        tr["type"] = "tool_result";
        tr["tool_use_id"] = msgs[i].tool_call_id;
        tr["content"] = msgs[i].content;
        jm["content"].push_back(tr);
        ++i;
      }
      --i; // outer loop will increment
    } else {
      jm["content"] = m.content;
    }
    body["messages"].push_back(jm);
  }

  if (!tools.empty()) {
    body["tools"] = json::array();
    for (const auto& t : tools) {
      json tj;
      tj["name"] = t.name;
      tj["description"] = t.description;
      tj["input_schema"] = t.to_json_schema()["parameters"];
      body["tools"].push_back(tj);
    }
  }

  auto [host, path] = split_url(cfg_.base_url, "/messages");
  httplib::Client cli(host);
  cli.set_read_timeout(120);

  httplib::Headers hdrs = { { "Content-Type", "application/json" } };
  if (!cfg_.api_key.empty()) {
    hdrs.emplace("x-api-key", cfg_.api_key);
  }
  hdrs.emplace("anthropic-version", "2023-06-01");

  std::string full;
  json acc = json::array();

  auto res = cli.Post(path,
                      hdrs,
                      body.dump(),
                      "application/json",
                      [&](const char* d, size_t n) -> bool {
                        return parse_sse_anthropic(std::string(d, n), full, acc,
                                                  callback);
                      });

  if (!res) {
    return make_assistant_msg(
      "[API error " + std::to_string(-1) + "]");
  }

  return build_msg(full, acc);
}

// ── helpers ────────────────────────────────────────────────────────

std::pair<std::string, std::string>
ApiModel::split_url(const std::string& url, const std::string& suffix)
{
  std::string u = url;
  std::string p = suffix;
  if (auto pos = u.find("://"); pos != std::string::npos) {
    auto host_start = pos + 3;
    if (auto slash = u.find('/', host_start); slash != std::string::npos) {
      p = u.substr(slash) + suffix;
      u = u.substr(0, slash);
    }
  }
  return { u, p };
}

bool
ApiModel::parse_sse(const std::string& chunk,
                    std::string& full,
                    json& acc,
                    ResponseCallback& cb)
{
  size_t pos = 0;
  while (pos < chunk.size()) {
    auto nl = chunk.find('\n', pos);
    std::string line = chunk.substr(pos, nl - pos);
    pos = (nl == std::string::npos) ? chunk.size() : nl + 1;
    if (line.empty() || line[0] == '\r') {
      continue;
    }
    if (line.rfind("data: ", 0) != 0) {
      continue;
    }
    std::string data = line.substr(6);
    if (data == "[DONE]") {
      continue;
    }
    try {
      auto j = json::parse(data);
      const auto& choices = j.value("choices", json::array());
      if (choices.empty()) {
        continue;
      }
      const auto& delta = choices[0].value("delta", json::object());
      if (delta.contains("content") && delta["content"].is_string()) {
        std::string t = delta["content"];
        full += t;
        if (cb)
          cb(t);
      }
      if (delta.contains("tool_calls")) {
        for (const auto& tc : delta["tool_calls"]) {
          int idx = tc.value("index", 0);
          while (static_cast<int>(acc.size()) <= idx) {
            acc.push_back(json::object());
          }
          auto& a = acc[idx];
          if (tc.contains("id"))
            a["id"] = tc["id"];
          if (tc.contains("function")) {
            if (tc["function"].contains("name")) {
              a["name"] = tc["function"]["name"];
            }
            if (tc["function"].contains("arguments")) {
              if (!a.contains("arguments"))
                a["arguments"] = "";
              a["arguments"] = a["arguments"].get<std::string>() +
                               tc["function"]["arguments"].get<std::string>();
            }
          }
        }
      }
    } catch (...) {
    }
  }
  return true;
}

bool
ApiModel::parse_sse_anthropic(const std::string& chunk,
                              std::string& full,
                              json& acc,
                              ResponseCallback& cb)
{
  size_t pos = 0;
  std::string current_event;
  while (pos < chunk.size()) {
    auto nl = chunk.find('\n', pos);
    std::string line = chunk.substr(pos, nl - pos);
    pos = (nl == std::string::npos) ? chunk.size() : nl + 1;

    if (line.rfind("event: ", 0) == 0) {
      current_event = line.substr(7);
      continue;
    }
    if (line.rfind("data: ", 0) != 0)
      continue;
    std::string data = line.substr(6);
    try {
      auto j = json::parse(data);
      if (current_event == "content_block_delta") {
        const auto& delta = j["delta"];
        if (delta.value("type", "") == "text_delta") {
          std::string t = delta["text"];
          full += t;
          if (cb)
            cb(t);
        } else if (delta.value("type", "") == "input_json_delta") {
          std::string partial = delta["partial_json"];
          if (!acc.empty()) {
            auto& a = acc.back();
            if (!a.contains("arguments"))
              a["arguments"] = "";
            a["arguments"] = a["arguments"].get<std::string>() + partial;
          }
        }
      } else if (current_event == "content_block_start") {
        const auto& block = j["content_block"];
        if (block.value("type", "") == "tool_use") {
          json a;
          a["name"] = block["name"];
          a["id"] = block["id"];
          a["arguments"] = "";
          acc.push_back(a);
        }
      }
    } catch (...) {
    }
  }
  return true;
}

common_chat_msg
ApiModel::build_msg(const std::string& full, const json& acc)
{
  common_chat_msg msg;
  msg.role = MessageRole::ASSISTANT;
  msg.content = full;
  for (const auto& a : acc) {
    common_chat_tool_call tc;
    tc.tool_name = a.value("name", "");
    tc.tool_args = a.value("arguments", "{}");
    tc.tool_call_id = a.value("id", "call_0");
    if (!tc.tool_name.empty())
      msg.tool_calls.push_back(std::move(tc));
  }
  return msg;
}

} // namespace zato
