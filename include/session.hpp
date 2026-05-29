#pragma once

#include "chat.hpp"
#include "model.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace zato {

namespace fs = std::filesystem;

// Persists conversation messages (JSON) and KV cache (binary) to disk,
// restoring context across restarts.
class SessionManager
{
public:
  SessionManager(std::shared_ptr<Model> model,
                 const fs::path& base_dir,
                 const std::string& name = "default")
    : model_(std::move(model))
  {
    dir_ = base_dir / ".zato" / "sessions" / name;
    fs::create_directories(dir_);
    msg_path_ = dir_ / "messages.json";
    cache_path_ = dir_ / "kv_cache.bin";
  }

  // Load previous session. Returns empty vector if no saved session exists.
  std::vector<common_chat_msg> load()
  {
    std::vector<common_chat_msg> messages;

    if (fs::exists(msg_path_)) {
      std::ifstream file(msg_path_);
      if (file.is_open()) {
        try {
          json j = json::parse(file);
          for (const auto& item : j) {
            messages.push_back(common_chat_msg::from_json_oaicompat(item));
          }
        } catch (...) {
          // Corrupted session file — start fresh
        }
      }
    }

    // KV cache loading is deliberately skipped: cross-session state
    // restoration via llama_memory_seq_rm + incremental decode can
    // cause attention corruption. The model re-encodes from scratch
    // on first turn — slower but reliable.
    (void)cache_path_;

    return messages;
  }

  // Save current session state.
  void save(const std::vector<common_chat_msg>& messages)
  {
    if (messages.empty()) {
      return;
    }

    json j = json::array();
    for (const auto& msg : messages) {
      j.push_back(msg.to_json_oaicompat());
    }

    std::ofstream file(msg_path_);
    if (file.is_open()) {
      file << j.dump(2) << "\n";
    }

    try {
      model_->save_cache(cache_path_.string());
    } catch (...) {
      // Cache save is best-effort
    }
  }

  [[nodiscard]] const fs::path& dir() const { return dir_; }

private:
  std::shared_ptr<Model> model_;
  fs::path dir_;
  fs::path msg_path_;
  fs::path cache_path_;
};

} // namespace zato
