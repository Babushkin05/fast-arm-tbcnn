#pragma once

#include <iostream>
#include <sstream>
#include <cstring>
#include <mutex>
#include <chrono>
#include <string>

namespace tbn {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4
};

class Logger {
private:
    static LogLevel global_level_;
    static std::mutex mutex_;

    LogLevel level_;
    std::string component_;

public:
    Logger(LogLevel level, const std::string& component)
        : level_(level), component_(component) {}

    static void set_global_level(LogLevel level);
    static LogLevel get_global_level();

    template<typename... Args>
    void log(const std::string& message) {
        if (level_ < global_level_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        std::string level_str;
        switch (level_) {
            case LogLevel::DEBUG:   level_str = "DEBUG"; break;
            case LogLevel::INFO:    level_str = "INFO "; break;
            case LogLevel::WARNING: level_str = "WARN "; break;
            case LogLevel::ERROR:   level_str = "ERROR"; break;
            case LogLevel::FATAL:   level_str = "FATAL"; break;
        }

        std::cout << "[" << level_str << "][" << component_ << "] " << message << std::endl;
    }
};

// Implementation in cpp file to avoid ODR violations
// extern LogLevel Logger::global_level_;
// extern std::mutex Logger::mutex_;

#define TBN_LOG_COMPONENT(name) \
    static tbn::Logger logger(tbn::LogLevel::INFO, name)

#define TBN_LOG_DEBUG(msg) \
    tbn::Logger(tbn::LogLevel::DEBUG, __func__).log(msg)

#define TBN_LOG_INFO(msg) \
    tbn::Logger(tbn::LogLevel::INFO, __func__).log(msg)

#define TBN_LOG_WARNING(msg) \
    tbn::Logger(tbn::LogLevel::WARNING, __func__).log(msg)

#define TBN_LOG_ERROR(msg) \
    tbn::Logger(tbn::LogLevel::ERROR, __func__).log(msg)

#define TBN_LOG_FATAL(msg) \
    do { \
        tbn::Logger(tbn::LogLevel::FATAL, __func__).log(msg); \
        std::exit(1); \
    } while(0)

} // namespace tbn

// Include implementation
// Static members defined in src/utils/logging.cpp