#pragma once

#include <stdio.h>
#include <numeric>
#include <iomanip>

#include "common.h"

class Player{
 public:
    std::string name;
    virtual std::pair<int,int> move(mtx<int>& board)=0;
};

class RandomPlayer: public Player{
 public:
    RandomPlayer():Player(){name="random";}
    virtual std::pair<int,int> move(mtx<int>& board);
};

class HumanPlayer: public Player{
 public:
    HumanPlayer():Player(){name="human";}
    virtual std::pair<int,int> move(mtx<int>& board);
};

std::pair<int,int> RandomPlayer::move(mtx<int>& board){
    std::vector<std::pair<int,int>> free_fields;
    for(int y=0;y<COL;y++){
        for(int x=0;x<ROW;x++){
            if(board[y][x]==0) free_fields.push_back({y,x});
        }
    }
    int y,x;
    std::tie(y,x) = free_fields[std::rand()%(ROW*COL)];
    return {y,x};
}

std::pair<int,int> HumanPlayer::move(mtx<int>& board){
    int x,y;
    do{
        std::cin>>y>>x;
    } while(board[y][x]!=0);
    return {y,x};
    //return y*ROW+x;
}



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
