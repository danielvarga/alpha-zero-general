#include <stdio.h>
#include <numeric>
#include <iomanip>
#include <iostream>

#include <tensorflow/c/c_api.h>
#include "cppflow/include/Model.h"
#include "cppflow/include/Tensor.h"

int main() {
    //set new shell environmental variable using putenv 
    char mypath[]="CUDA_VISIBLE_DEVICES=3"; 
    putenv( mypath );
 
    printf("Hello from TensorFlow C library version %s\n", TF_Version());
    Model model("model.pb");
    model.init();

    auto input = new Tensor(model, "input");
    auto output  = new Tensor(model, "prediction");

    std::cout<<"============== INIT DONE =============\n";
    for(int i=0;i<3;i++){
      //std::vector<double> data(10);
	std::vector<float> data(10, 1);
	input->set_data(data);

	model.run(input, output);
	std::cout<<"Gooood\n";
	std::cout<<output->get_data<float>()[0]<<std::endl;
	std::cout<<output->get_data<float>()[1]<<std::endl;
	std::cout<<output->get_data<float>()[2]<<std::endl;
	std::cout << std::endl;
    }
    return 0;
}
