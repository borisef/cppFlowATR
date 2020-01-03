#include <iostream>
#include <future>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>

using namespace std;
using namespace std::chrono;



class Runner
{
    public:
    
    virtual int Operate() = 0;
	
	std::future<int> result;
    
    int countR1 = 0;
    int countR2 = 0;
    
     void Run1()
    {
         cout<<"----Run1 begin----"<<endl;
        float milliseconds = 500;
        std::this_thread::sleep_for(std::chrono::milliseconds((uint16_t)milliseconds));
        std::cout << "R1: Waited sec:" << (milliseconds/1000.0)<<endl;
         cout<<"----Run1 end----"<<endl;
        ++countR1;
    }
    virtual void Run2()
    {
        cout<<"----Run2 begin----"<<endl;
        float milliseconds = 1344;
        std::this_thread::sleep_for(std::chrono::milliseconds((uint16_t)milliseconds));
        std::cout << "R2: Waited sec:" << (milliseconds/1000.0)<<endl;
        cout<<"----Run2 end----"<<endl;
        ++countR2;
    }

};

class RunnerHandler:public Runner
{
    public: 
    bool m_isBusy = false;
    std::mutex m_mutex;
    int Operate()
    {
        m_isBusy = true;
        Run2();
        m_isBusy = false;
        return 0;
    }

};

void OperateAPI(Runner* rrr, int step)
	{
		if(rrr != nullptr){
			RunnerHandler* handle = (RunnerHandler*)rrr;
			// Copy output data
			// ...
            handle->Run1();
			if (!handle->m_isBusy){
                
                std::cout<< "Run step" << step<<endl; 
				// Copy input data
				handle->result = std::async(std::launch::async, &RunnerHandler::Operate, handle);
			}
		}
	}


int main()
{
    
    cout<<"Hi"<<endl;
    Runner* myrunner = (Runner*) new RunnerHandler();

for (int i = 0;i<10;i++)
    OperateAPI(myrunner,i); 

cout<<"Run1 count"<<myrunner->countR1<<endl;
cout<<"Run2 count"<<myrunner->countR2<<endl;
    

}