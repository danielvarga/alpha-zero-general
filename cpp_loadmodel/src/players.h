#pragma once

#include <stdio.h>
#include <numeric>
#include <iomanip>
#include <iostream>

#include "common.h"
#include "heuristic.h"
class Player{
 public:
    std::string name;
    virtual std::pair<int,int> move(mtx<int>& board, int player, bool log)=0;
};

class RandomPlayer: public Player{
 public:
    RandomPlayer():Player(){name="random";}
    std::pair<int,int> move(mtx<int>& board, int player, bool log);
};

class HumanPlayer: public Player{
 public:
    HumanPlayer():Player(){name="human";}
    std::pair<int,int> move(mtx<int>& board, int player, bool log);
};

class PolicyPlayer: public Player{
 public:
    PolicyPlayer();
    std::pair<int,int> move(mtx<int>& board, int player, bool log);
 public:
    Tensor *board_tens = nullptr, *curPlayer=nullptr, *dropout=nullptr, *probs=nullptr, *v=nullptr;
    Heuristic h;
    Model* model=nullptr;
};

std::pair<int,int> RandomPlayer::move(mtx<int>& board, int player, bool log){
    std::vector<std::pair<int,int>> free_fields;
    for(int y=0;y<COL;y++){
        for(int x=0;x<ROW;x++){
            if(board[y][x]==0) free_fields.push_back({y,x});
        }
    }
    int y,x;
    std::tie(y,x) = free_fields[std::rand()%(ROW*COL)];
    std::cout<<y<<" "<<x<<std::endl;
    return {y,x};
}

std::pair<int,int> HumanPlayer::move(mtx<int>& board, int player, bool log){
    int x,y;
    do{
        std::cin>>y>>x;
    } while(board[y][x]!=0);
    printf("\033[1A"); // Correct the ENTER
    return {y,x};
    //return y*ROW+x;
}

PolicyPlayer::PolicyPlayer() : Player(){
    name="policy";
    char mypath[]="CUDA_VISIBLE_DEVICES=1"; 
    putenv(mypath );

    model = new Model("../model.pb");
    model->restore("../checkpoint/train.ckpt");

    board_tens= new Tensor(*model, "input_boards");
    curPlayer = new Tensor(*model, "graph2/curPlayer");
    dropout = new Tensor(*model, "graph2/dropout");
    
    probs  = new Tensor(*model, "graph2/prob");
    v  = new Tensor(*model, "graph2/v");
    
    int x,y;
    mtx<int> board;
    make_zero(board);
    std::tie(y,x) = move(board, 1, true);
    std::cout<<"============== INIT DONE =============\n";
}

std::pair<int,int> PolicyPlayer::move(mtx<int>& board, int player, bool log){
    std::vector<float> board_data = h.get_flat_layers<float>(board);
    std::vector<float> player_data = std::vector<float>(1, player);
    
    board_tens->set_data(board_data);
    curPlayer->set_data(player_data);
    
    model->run({board_tens, curPlayer}, {probs, v});
    auto result_probs = probs->get_data<float>();
    for(int y=0;y<COL;y++){
        for(int x=0;x<ROW;x++){
            if (board[y][x] !=0){
                result_probs[y*ROW+x]=0.0;
            }
        }
    }
    auto max_result = std::max_element(result_probs.begin(), result_probs.end());
    int argmax = std::distance(result_probs.begin(),max_result); // result-result_probs.begin()
    return {argmax/ROW, argmax%ROW};
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
    
    std::vector<float> board_data(432, 0.0);
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

    Heuristic h;
    PolicyPlayer p;
    mtx<int> myboard;
    make_zero(myboard);
    
    std::cout<<"================ INIT DONE ================"<<std::endl;
    for (int i=0;i<10;i++){
      print_mtx(myboard);
      int actPlayer = 1-2*(i%2);
      int y,x;
      std::tie(y,x)=p.move(myboard, actPlayer, true);
      myboard[y][x]=actPlayer;
      std::cout<<actPlayer<<" "<<x<<" "<<y<<std::endl;
    }
}
