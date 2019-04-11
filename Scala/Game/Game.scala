import Interface._
import Board._

/**
 * Main code of the program
 */
object Game
{
  def main (args: Array[String])
  {
    // Title
    Interface.printWelcome;
    /*
     * Start or exists the code
     * 
     * If the player decides to start the game, it is first needed to specify
     * the game difficulty. 
     */
    val action = Interface.pickAction 

    if(action == 1) createGameI(Interface.pickDifficulty);
    else Interface.exitGame;
    // After being specified the difficulty, the game can finally start
  }

  /**
   *  Creates the game according to the difficulty settings.
   */
  def createGame(difficulty: Any => Int)
  {
    Interface.printBoard(Board.seedBoard(difficulty, Board.initBoard(4)))  
  }

  /**
   *  Print the controls and the State of each round.
   */
  def printState(lives: Int, points: Int)
  {
    Interface.printControls;
    Interface.printPoints(points)
    Interface.printLives(3, lives)
  }
}
