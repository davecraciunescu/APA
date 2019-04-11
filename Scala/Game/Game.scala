import Interface._
import Board._
import Color._

/**
 * Main code of the program
 */
object Game
{
  def main (args: Array[String])
  {
    // Welcome screen.
    // Ask for action.
    //  - Play.
    //    - Pick difficulty.
    //    - Generate lives.
    //
    //    - Generate board.
    //    - Seed board.
    //    
    //    - Play game.
    //      - Check can play. 
    //        - Yes 
    //          - Print board on screen.
    //          - Print info on screen.
    //          - Ask for action.
    //            - Check move.
    //              - Move board.
    //              - Join board.
    //              - Seed board.
    //            - Quit game.
    //              - Quit.
    //        - No
    //          - Print end screen.
    //          - Take life away.
    //    - Want to play again.
    //      - Yes
    //        - Check can play
    //          - Yes
    //            - Play again
    //          - No
    //            - Can't play
    //      - No
    //        - Quit
    //  - Quit
    //

    // Title
    Interface.printWelcome;

    val action = Interface.pickAction

    if (action == 1) 
    {
      val difficulty = pickDifficulty; 
      val      lives = 3
      val   theBoard = createBoard(difficulty) 
    }  
    else Interface.exitGame

    // After being specified the difficulty, the game can finally start
    createGame(difficulty)
  }


  /**
   *  Execute the game.
   */ 
  def playGame(lives: Int, diff: Int, board: List[Int]) =
  {
    if (lives > 0)
    {
      if (!Board.isEmpty(board)) 
      {
        Interface.printBoard(board)
        printState(lives, Board.getPoints(board))
      
        val move = Interface.pickMove

        if (move.matches("[Qq]") println("Thanks for playing :)".green.bold)
        else
        {
          // Invoke board value movement.
          // Join values in board.
          // playGame(lives, diff, newBoard)
        }
      } 
      else 
      {  
        printEndScreen(Board.getPoints(board))

        if (Interface.playAgain) playGame(lives - 1, createBoard(diff))
        else println("Thanks for playing :)".green.bold)
      }
    }
    else 
    {  
      printEndScreen(Board.getPoints(board))
      println("No more lives left".red.bold.blink)
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
      case 3 => 17
    }

    Board.seedBoard(Board.initBoard(size))
  }
}
