#pragma once

#include <string>

namespace zato::ansi {

// --- escape codes (use directly with <<) ---
inline constexpr const char* kReset  = "\033[0m";
inline constexpr const char* kBold   = "\033[1m";
inline constexpr const char* kDim    = "\033[2m";

inline constexpr const char* kRed     = "\033[31m";
inline constexpr const char* kGreen   = "\033[32m";
inline constexpr const char* kYellow  = "\033[33m";
inline constexpr const char* kBlue    = "\033[34m";
inline constexpr const char* kMagenta = "\033[35m";
inline constexpr const char* kCyan    = "\033[36m";
inline constexpr const char* kWhite   = "\033[37m";

inline constexpr const char* kBrightRed     = "\033[91m";
inline constexpr const char* kBrightGreen   = "\033[92m";
inline constexpr const char* kBrightYellow  = "\033[93m";
inline constexpr const char* kBrightBlue    = "\033[94m";
inline constexpr const char* kBrightMagenta = "\033[95m";
inline constexpr const char* kBrightCyan    = "\033[96m";

// --- convenience: wrap a string with colour + reset ---
inline std::string red(std::string_view s)   { return std::string(kRed) + std::string(s) + kReset; }
inline std::string green(std::string_view s) { return std::string(kGreen) + std::string(s) + kReset; }
inline std::string yellow(std::string_view s){ return std::string(kYellow) + std::string(s) + kReset; }
inline std::string cyan(std::string_view s)  { return std::string(kCyan) + std::string(s) + kReset; }
inline std::string bold(std::string_view s)  { return std::string(kBold) + std::string(s) + kReset; }

} // namespace zato::ansi
