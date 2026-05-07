#pragma once
#include <string>

namespace ba {

struct Status {
    bool ok = true;
    std::string message;

    static Status OK() {
        return {true, {}};
    }

    static Status Error(std::string m) {
        return {false, std::move(m)};
    }

    explicit operator bool() const {
        return ok;
    }
};

} // namespace ba
