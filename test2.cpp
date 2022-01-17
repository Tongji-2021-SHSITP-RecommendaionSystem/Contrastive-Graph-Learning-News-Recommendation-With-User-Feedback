#include <iostream>
#include <unistd.h>

using namespace std;

int main(int argc, char *argv[])
{
    if(argc!=2){
        cout<<"arg error"<<endl;
        exit(0);
    }
    while(1)
    {
    	cout<<argv[1]<<endl;
    	sleep(2);
    }
    return 0;
}
