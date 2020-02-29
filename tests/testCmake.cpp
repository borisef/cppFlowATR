#include <iostream>

#include <chrono>
#include <thread>

using namespace std;

int main()
{
#ifdef WIN32
    cout << "#ifdef WIN32  = YES" << endl;
#else
    cout << "#ifdef WIN32  = NO" << endl;
#endif

#ifdef TEST_MODE
    cout << "#ifdef TEST_MODE  = YES" << endl;
#else
    cout << "#ifdef TEST_MODE  = NO" << endl;
#endif

#ifdef OS_WINDOWS
    cout << "#ifdef OS_WINDOWS  = YES" << endl;
#else
    cout << "#ifdef OS_WINDOWS  = NO" << endl;
#endif

#ifdef OS_LINUX
    cout << "#ifdef OS_LINUX  = YES" << endl;
#else
    cout << "#ifdef OS_LINUX  = NO" << endl;
#endif

#ifdef JETSON
    cout << "#ifdef JETSON  = YES" << endl;
#else
    cout << "#ifdef JETSON  = NO" << endl;
#endif

    return 0;
}