import Interface._
import Board._

/**
 * Main code of the program
 */
object Game
{
  def main (args: Array[String])
  {
    printState(2, 200)

    createGame(2)
  }

  /**
   *  Creates the game according to the difficulty settings.
   */
  def createGame(difficulty: Int)
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
