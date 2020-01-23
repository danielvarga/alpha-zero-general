#include <stdio.h>
#include <numeric>
#include <iomanip>
#include <iostream>

#include <tensorflow/c/c_api.h>
#include "cppflow/include/Model.h"
#include "cppflow/include/Tensor.h"
//#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iterator>

void run_toy(){
    //set new shell environmental variable using putenv 
    char mypath[]="CUDA_VISIBLE_DEVICES=0"; 
    putenv(mypath );
 
    printf("Hello from TensorFlow C library version %s\n", TF_Version());
    Model model("../model.pb");
    model.restore("../checkpoint/train.ckpt");

    auto inp1 = new Tensor(model, "a");
    auto inp2 = new Tensor(model, "b");
    auto output  = new Tensor(model, "out");

    std::cout<<"============== INIT DONE =============\n";
    std::vector<float> data(10, 1);
    inp1->set_data(data);
    inp2->set_data(data);
    
    model.run({inp1, inp2}, output);
    std::cout<<"Gooood\n";
    std::cout<<output->get_data<float>()[0]<<std::endl;
    std::cout << std::endl;
}

void advanced(){
    Model model("../model.pb");
    model.restore("../checkpoint/train.ckpt");

    auto board = new Tensor(model, "input_boards");
    auto curPlayer = new Tensor(model, "graph2/curPlayer");
    auto dropout = new Tensor(model, "graph2/dropout");

    auto probs  = new Tensor(model, "graph2/prob");
    auto v  = new Tensor(model, "graph2/v");
    
    std::vector<float> board_data(432, 1.0);
    std::vector<float> player_data(1, 1.0);
    
    board->set_data(board_data);
    curPlayer->set_data(player_data);
    
    model.run({board, curPlayer}, {probs, v});
    auto result_v = v->get_data<float>();
    auto result_probs = probs->get_data<float>();
    auto max_result = std::max_element(result_probs.begin(), result_probs.end());
    int argmax = std::distance(result_probs.begin(),max_result); // result-result_probs.begin()
    std::cout<<result_v.size()<<" "<<result_probs.size()<<std::endl;
    std::cout<<"Max: "<<(*max_result)<<" "<<argmax<<std::endl;

    std::cout<<"================ INIT DONE ================"<<std::endl; 
    for (int i=0;i<10;i++){
          std::vector<float> board_data(432, 1.0);
	  std::vector<float> player_data(1, 1.0);
    
	  board->set_data(board_data);
	  curPlayer->set_data(player_data);
    
	  model.run({board, curPlayer}, {probs, v});
	  std::cout<<v->get_data<float>()[0]<<std::endl;
    }
}

int main() {
  //example();
  //run_toy();
  advanced();
    return 0;
}
