#include<iostream>

class Game;

class Game_AI
{

     int response;
     int apocalypse;
    char control;

public:
    int max;
    int win;
    int plus;
    int score;
    int grid[4][4];
    int bgrid[4][4];

    Game_AI(): score(0), plus(0), win(2048), max(0), response(0), apocalypse(1) {}
    
    void logic_flow(Game*);
    void game_end_check(Game*);
    void key_press();
    void start_grid();
    void update_grid();
    void fill_space();
    void spawn();
    void find_greatest_tile();
    void backup_grid();
    void undo();

    int full();
    int block_moves();
};

class Game : public Game_AI
{
    char option;
    std::string name;
    
public:
    void display_grid();
    void display_help_screen();
    void display_win_screen();
    void display_loser_screen();
    char display_try_again_screen(int);

};

int main()
{
    Game exec;
    
    srand(time(NULL));

    exec.start_grid();

    while(1)
    {
        exec.display_grid();
        exec.key_press();
        exec.logic_flow(&exec);
        exec.game_end_check(&exec);
    };
    return 0;
}


