#include "unicode.h"
#include <iostream>
#include <string>
#include <vector>

int main() {
    const std::string tiny = "\\d{1,3}(?=(?:\\d{3})*\\b)";
    const std::string afmoe = "\\p{AFMoE_digits}";
    struct test_case {
        std::string text;
        std::vector<std::string> expected;
    };
    const std::vector<test_case> cases = {
        {"  46", {"  ", "46"}},
        {" 1234567", {" ", "1", "234", "567"}},
        {"x: 12, 3456", {"x: ", "12", ", ", "3", "456"}},
        {"no digits", {"no digits"}},
        {"123", {"123"}},
        {"1480s", {"1480s"}},
        {"1970s and 1930s", {"1970s and 1930s"}},
        {"12_name", {"12_name"}},
        {"1480s  46", {"1480s  ", "46"}},
        {"2007\u200c", {"2", "007", "\u200c"}},
        {"2007\u200d", {"2", "007", "\u200d"}},
        {"2007\u0301", {"2007\u0301"}},
        {"  ২০০৭\u200c", {"  ", "২", "০০৭", "\u200c"}},
        {"  २००७\u200d", {"  ", "२", "००७", "\u200d"}},
        {"", {}},
    };
    for (const auto & test : cases) {
        if (unicode_regex_split(test.text, {tiny}, false) != test.expected) {
            std::cerr << "Tiny Aya prefix isolation failed\n";
            return 1;
        }
    }
    if (unicode_regex_split("  46", {afmoe}, false) != std::vector<std::string>{"  46"}) {
        std::cerr << "AFMoE behavior changed\n";
        return 1;
    }
    std::cout << cases.size() << " Tiny Aya cases and AFMoE preservation passed\n";
}
