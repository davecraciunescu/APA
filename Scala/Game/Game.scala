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

    val action: Interface.pickAction

    if (action == 1) playGame()
    else Interface.exitGame

    // After being specified the difficulty, the game can finally start
    createGame(diff)
  }

  /**
   *  Creates the game according to the difficulty settings.
   */
  def createGame(difficulty: Int): List[Int] =
  {
    val size: Int = difficulty match
    {
      case 1 => 4 
      case 2 => 9
      case 3 => 14
      case 3 => 17
    }

    Board.seedBoard(difficulty, Board.initBoard(size))
  }

  def playGame() =
  {
    val board = createGame(Interface.pickDifficulty)

  

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
