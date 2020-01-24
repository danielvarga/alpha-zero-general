#include <tensorflow/c/c_api.h>
#include "Model.h"
#include "Tensor.h"

#include "model.h"

#include "common.h"
#include "heuristic.h"
#include "gobanggame.h"


int main() {
    Heuristic h;
    GobangGame game;
    game.play(true);

    mtx<int> board = game.init();
    auto layers = h.get_layers<double>(board);
    for(int k=0;k<layers.size();k++){
        std::cout<<"Layer "<<k<<std::endl;
        print_mtx(layers[k], k==0?5:0);
    }
    return 0;
}
