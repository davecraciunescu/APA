import Interface._
import Board._
import Color._
import Movements._

/**
 * Main code of the program
 */
object Game
{
  def main (args: Array[String])
  {
    // Title
    Interface.printWelcome;

    val action = Interface.pickAction

    if (action == 1) 
    {
      val difficulty = pickDifficulty; 
      val      lives = 3
      val     points = 0
      playGame(lives, points, difficulty, createBoard(difficulty))
    }  
    else Interface.exitGame
  }

  /**
   *  Execute the game.
   */ 
  def playGame(lives: Int, points: Int, diff: Int, board: List[Int]): Unit =
  {
    if (lives > 0)
    {
      if (!Board.isEmpty(board)) 
      {
        Interface.printBoard(board)
        printState(lives, Board.getPoints(board))
      
        val move = Interface.pickMove

        if (move.matches("[Qq]")) println("Thanks for playing :)".green.bold)
        else
        {
          val pts = Board.getPoints(board)
          val newBoard = Movements.mover(move, board, math.sqrt(board.size).toInt)            
          val seeded   = Board.seedBoard(newBoard)
          
          playGame(lives, pts, diff, seeded)
        }
      } 
      else 
      {
        printEndScreen(points)
        println;
  
        if (Interface.playAgain) playGame(lives - 1, 0, diff, createBoard(diff))
        else println("Thanks for playing :)".green.bold)
      }
    }
    else 
    {  
      printEndScreen(Board.getPoints(board))
      println("No more lives left".red.bold.blink)
      println;
    }
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
  
  /**
   *  Creates the game according to the difficulty settings.
   */
  def createBoard(difficulty: Int): List[Int] =
  {
    val size: Int = difficulty match
    {
      case 1 => 4 
      case 2 => 9
      case 3 => 14
      case 4 => 17
    }

    Board.seedBoard(Board.initBoard(size))
  }
}
