#pragma once

#include "chat.hpp"
#include "imodel.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace zato {

class ApiModel final : public IModel
{
public:
  struct Config
  {
    std::string base_url;
    std::string api_key;
    std::string model;
  };

  explicit ApiModel(Config cfg);

  common_chat_msg generate(const std::vector<common_chat_msg>& messages,
                           const std::vector<common_chat_tool>& tools,
                           ResponseCallback callback) override;

private:
  common_chat_msg generate_openai(const std::vector<common_chat_msg>& msgs,
                                  const std::vector<common_chat_tool>& tools,
                                  ResponseCallback callback);

  common_chat_msg generate_anthropic(const std::vector<common_chat_msg>& msgs,
                                     const std::vector<common_chat_tool>& tools,
                                     ResponseCallback callback);

  static std::pair<std::string, std::string>
  split_url(const std::string& url, const std::string& suffix);

  static bool parse_sse(const std::string& chunk, std::string& full,
                        nlohmann::json& acc, ResponseCallback& cb);

  static bool parse_sse_anthropic(const std::string& chunk, std::string& full,
                                  nlohmann::json& acc, ResponseCallback& cb);

  static common_chat_msg build_msg(const std::string& full,
                                   const nlohmann::json& acc);

  // Returns a connected client (reuses cached connection when possible)
  httplib::Client& get_client(const std::string& host);

  Config cfg_;
  bool is_anthropic_;
  std::unique_ptr<httplib::Client> cli_;
  std::string cli_host_; // track which host cli_ is connected to
};

} // namespace zato
