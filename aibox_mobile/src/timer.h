#ifndef _AI_BOX_TIMER_H_
#define _AI_BOX_TIMER_H_ 

#include <chrono>

namespace AIBox {

class Timer {
public:
    Timer() : start_time_point(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_time_point = std::chrono::high_resolution_clock::now();
    }

    double elapsed() {
        auto end_time_point = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_time = end_time_point - start_time_point;
        double duration = elapsed_time.count();
        reset();
        return duration;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
};

} // namespace AIBox

#endif // _AI_BOX_TIMER_H_