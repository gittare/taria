#include <iostream>

extern "C" {
    // Basic stub indicating the backend engine is linked
    void taria_engine_init() {
        std::cout << "Taria Backend Engine Initialized" << std::endl;
    }
}
