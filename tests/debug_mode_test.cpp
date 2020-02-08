#include <iostream>

int main(){
    bool success = false;

    #ifdef TEST_MODE
    success = true;
    #endif

    std::cout << "Test ";
    std::cout << (success ? "Succeeded":"Failed") << std::endl;

    return 0;
}