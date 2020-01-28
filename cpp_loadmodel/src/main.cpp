#include <tensorflow/c/c_api.h>
#include "Model.h"
#include "Tensor.h"

#include "players.h"

#include "common.h"
#include "heuristic.h"
#include "gobanggame.h"


int main() {
    Heuristic h;
    GobangGame game;
    char mypath[]="CUDA_VISIBLE_DEVICES=1"; 
    putenv(mypath );
    
    Player* p1 = new HumanPlayer();
    Player* p2 = new RandomPlayer();
    Player* p3 = new PolicyPlayer();
    game.arena(p1,p3, true);
    
    // === Print heur mtx ===
    mtx<int> board = game.init();
    auto layers = h.get_layers<double>(board);
    for(int k=0;k<layers.size();k++){
        std::cout<<"Layer "<<k<<std::endl;
        print_mtx(layers[k], k==0?5:0);
    }
    return 0;
}
