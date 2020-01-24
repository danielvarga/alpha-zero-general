#include <stdio.h>

#pragma once

typedef std::pair<int, int>(*FunctPtr)(mtx<int>&);

class GobangGame{
 public:
    mtx<int> init();
    void move(const int action, mtx<int>& board, const int curPlayer);
    void display(mtx<int>& board, bool end);
    void play(bool first);
    void arena(FunctPtr player1, FunctPtr player2);
};

struct bcolors{
 public:
    const static std::string WARNING;
    const static std::string FAIL;
    const static std::string ENDC;
};

const std::string bcolors::WARNING = "\033[93m";
const std::string bcolors::FAIL = "\033[91m";
const std::string bcolors::ENDC = "\033[0m";

mtx<int> GobangGame::init(){
    mtx<int> table;
    make_zero(table);
    return table;
}

void GobangGame::move(const int action, mtx<int>& board, const int curPlayer){
    int y = action/ROW;
    int x = action%ROW;
    board[y][x]=curPlayer;
}

void GobangGame::arena(FunctPtr player1, FunctPtr player2){
    mtx<int> board = init();
    int y,x;
    for(int i=0;i<COL*ROW/2;i++){
        std::tie(y,x)=player1(board);
        board[y][x]=1;
        display(board, false);

        std::tie(y,x)=player2(board);
        board[y][x]=1;
        display(board, false);
    }
}

void GobangGame::play(bool first){
    mtx<int> board = init();
    for(int i=0;i<COL*ROW/2;i++){
        int x,y;
        if(first){
            std::cin>>y>>x;
            board[y][x]=1;
            display(board, false);
            
            std::cin>>y>>x;//##
            board[y][x]=-1;
        }
        else{
            std::cin>>y>>x;//##
            board[y][x]=1;
            display(board, false);

            std::cin>>y>>x;
            board[y][x]=-1;
        }
        display(board, false);
    }
    display(board, true);
}

void GobangGame::display(mtx<int>& board, bool end){
    printf("                           \n");
    printf("  === Gobang Game ===\n");
    printf("  ");
    for(int i=0;i<COL;i++){
        std::string num = std::to_string(i);
        num = num.substr(num.size()-1,num.size());
        printf("%s ",num.c_str());
    }
    printf("\n");
    
    printf(" +========================+\n");
    for(int x =0;x<ROW;x++){
        printf("%d|",x);
        for(int y =0;y<COL;y++){
            int piece = board[y][x];
            if(piece>0){
                printf("%s%s%s", bcolors::WARNING.c_str(), "o ", bcolors::ENDC.c_str());
            }
            else if(piece<0){
                printf("%s%s%s", bcolors::FAIL.c_str(), "x ", bcolors::ENDC.c_str());
            }
            else{
                printf("  ");
            }
        }
        printf("|\n");
    }
    printf(" +========================+\n");
    if (!end) printf("\033[%dA",ROW+6);
    printf("                           \n");
    printf("\033[1A");
}





