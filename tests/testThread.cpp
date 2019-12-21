#include <iostream>
#include <future>
using namespace std;

void outStam(void)
{

}

class Runner
{
    public:
    bool m_isBusy = false;
    virtual void Operate() = 0;
	
	std::future<void> result;
    int countR1 = 0;
    int countR2 = 0;
    




    
     void Run1()
    {
        int L = 1000;
        for (int i=0;i<L;i++)
        {
            //like pause
        }
        cout<<"****Run1****"<<endl;
        ++countR1;
    }
    virtual void Run2()
    {
        int L = 500000;
        for (int i=0;i<L;i++)
        {
            //like pause
        }
        cout<<"----Run2----"<<endl;
        ++countR2;
    }

};

class RunnerHandler:public Runner
{
    public: 
    void Operate()
    {
        m_isBusy = true;
        Run2();
        m_isBusy = false;
    }

    public:
    
    public: 
        // void Stam(int a)
        // {


        // }
        // void Run2()
        // {
        //     cout<<"Hadler Run2"<<endl;
            
        //     Runner::Run2();
            
        // }
        // void RunMe(int s)
        // {
        //     cout<<" * * *"<<endl<<"Run in handler"<<endl;
        //     Run1();
        //     Run2();
        //     cout<<"Finished Running step " <<s<<endl;

        // }

};

void OperateAPI(Runner* rrr, int step)
	{
		if(rrr != nullptr){
			RunnerHandler* handle = (RunnerHandler*)rrr;
			// Copy output data
			// ...
            handle->Run1();
			if (!handle->m_isBusy){
                
				// Copy input data
				handle->result = std::async(std::launch::async, &RunnerHandler::Operate, handle);
			}
		}
	}


int main()
{
    
    cout<<"Hi"<<endl;
    Runner* myrunner = (Runner*) new RunnerHandler();

for (int i = 0;i<1000;i++)
    OperateAPI(myrunner,i); 

cout<<"Run1 count"<<myrunner->countR1<<endl;
cout<<"Run2 count"<<myrunner->countR2<<endl;
    

}