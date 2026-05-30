#include "api.hpp"

#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>

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
  auto& cli = get_client(host);

  httplib::Headers hdrs = { { "Content-Type", "application/json" } };
  if (!cfg_.api_key.empty()) {
    hdrs.emplace("Authorization", "Bearer " + cfg_.api_key);
  }

  std::string full;
  json acc = json::array();

  auto res =
    cli.Post(path,
             hdrs,
             body.dump(),
             "application/json",
             [&](const char* d, size_t n) -> bool {
               return parse_sse(std::string(d, n), full, acc, callback);
             });

  if (!res) {
    std::cerr << "\r  [API fail: " << httplib::to_string(res.error()) << "]"
              << '\n';
    return make_assistant_msg(
      "[API error: " + std::string(httplib::to_string(res.error())) + "]");
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
  bool has_system = false;
  for (size_t i = 0; i < msgs.size(); ++i) {
    const auto& m = msgs[i];
    if (m.role == MessageRole::SYSTEM) {
      if (!has_system) {
        body["system"] = m.content;
        has_system = true;
      } else {
        // Additional system messages (memory summaries) → user message
        json jm;
        jm["role"] = "user";
        jm["content"] = m.content;
        body["messages"].push_back(jm);
      }
      continue;
    }
    json jm;
    jm["role"] = role_to_string(m.role);
    if (!m.tool_calls.empty()) {
      jm["content"] = json::array();
      if (!m.content.empty()) {
        jm["content"].push_back({ { "type", "text" }, { "text", m.content } });
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
  auto& cli = get_client(host);

  httplib::Headers hdrs = { { "Content-Type", "application/json" } };
  if (!cfg_.api_key.empty()) {
    hdrs.emplace("x-api-key", cfg_.api_key);
  }
  hdrs.emplace("anthropic-version", "2023-06-01");

  std::string full;
  json acc = json::array();

  std::string raw_sse;
  auto res = cli.Post(path,
                      hdrs,
                      body.dump(),
                      "application/json",
                      [&](const char* d, size_t n) -> bool {
                        raw_sse.append(d, n);
                        return parse_sse_anthropic(
                          std::string(d, n), full, acc, callback);
                      });

  if (!res) {
    std::cerr << "\r  [API fail: " << httplib::to_string(res.error()) << "]"
              << '\n';
    return make_assistant_msg(
      "[API error: " + std::string(httplib::to_string(res.error())) + "]");
  }

  auto msg = build_msg(full, acc);
  if (msg.content.empty() && msg.tool_calls.empty() && !raw_sse.empty()) {
    std::cerr << "  [empty] SSE:"
              << raw_sse.substr(0, std::min(raw_sse.size(), size_t(300)))
              << "\n";
  }
  return msg;
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
        if (cb) {
          cb(t);
        }
      }
      if (delta.contains("tool_calls")) {
        for (const auto& tc : delta["tool_calls"]) {
          int idx = tc.value("index", 0);
          while (static_cast<int>(acc.size()) <= idx) {
            acc.push_back(json::object());
          }
          auto& a = acc[idx];
          if (tc.contains("id")) {
            a["id"] = tc["id"];
          }
          if (tc.contains("function")) {
            if (tc["function"].contains("name")) {
              a["name"] = tc["function"]["name"];
            }
            if (tc["function"].contains("arguments")) {
              if (!a.contains("arguments")) {
                a["arguments"] = "";
              }
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
  // Parse SSE lines — handles both "event: + data:" and bare "data:" formats
  size_t pos = 0;
  std::string event_type;
  while (pos < chunk.size()) {
    auto nl = chunk.find('\n', pos);
    std::string line = chunk.substr(pos, nl - pos);
    pos = (nl == std::string::npos) ? chunk.size() : nl + 1;

    if (line.rfind("event: ", 0) == 0) {
      event_type = line.substr(7);
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
      // Use JSON "type" field as fallback when no event: line present
      std::string type = j.value("type", event_type);
      const auto& delta = j.value("delta", json::object());

      if (type == "content_block_delta") {
        if (delta.value("type", "") == "text_delta") {
          std::string t = delta.value("text", "");
          if (!t.empty()) {
            full += t;
            if (cb) {
              cb(t);
            }
          }
        } else if (delta.value("type", "") == "input_json_delta") {
          std::string partial = delta.value("partial_json", "");
          if (!partial.empty() && !acc.empty()) {
            auto& a = acc.back();
            if (!a.contains("arguments")) {
              a["arguments"] = "";
            }
            a["arguments"] = a["arguments"].get<std::string>() + partial;
          }
        }
      } else if (type == "content_block_start") {
        const auto& block = j.value("content_block", json::object());
        if (block.value("type", "") == "tool_use") {
          json a;
          a["name"] = block.value("name", "");
          a["id"] = block.value("id", "");
          a["arguments"] = "";
          acc.push_back(a);
        }
      } else if (type == "message_stop" || type == "message_delta") {
        // End of message — stop processing
        break;
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
  // Never return empty — fallback for silent API failures
  if (msg.content.empty() && acc.empty()) {
    msg.content = "[no response — context may be too long, try /clear]";
  }
  for (const auto& a : acc) {
    common_chat_tool_call tc;
    tc.tool_name = a.value("name", "");
    tc.tool_args = a.value("arguments", "{}");
    tc.tool_call_id = a.value("id", "call_0");
    if (!tc.tool_name.empty()) {
      msg.tool_calls.push_back(std::move(tc));
    }
  }
  return msg;
}

httplib::Client&
ApiModel::get_client(const std::string& host)
{
  if (!cli_ || cli_host_ != host) {
    cli_ = std::make_unique<httplib::Client>(host);
    cli_->set_connection_timeout(10);
    cli_->set_read_timeout(60);
    cli_->set_write_timeout(30);
    cli_->set_keep_alive(true);
    cli_host_ = host;
  }
  return *cli_;
}

} // namespace zato
