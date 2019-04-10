import Color._
import scala.io.StdIn.{readLine, readInt}
import scala.language.implicitConversions

/**
 *  Graphical interface and I/O interaction with user.
 *
 *      @author: Dave E. Craciunescu. Pablo Acereda Garcia
 *        @date: 2019.04.08
 *
 *        @todo: 
 *
 *   @changelog:
 *    -- 2019.04.08 -- Dave E.
 *      -- Add welcome screen ASCII Art.
 *      -- Add color coding.
 *    -- 2019.04.09 -- Dave E.
 *      -- Add game menus and validation system.
 *    -- 2019.04.10 -- Dave E.
 *      -- Add grid printing format.
 *      -- Add round printing hook.
 *
 *   @knownBugs:
 *    -- Color does not work in Windows terminal.
 *    -- string.isBlank does not execute in Windows envionments.
 */
object Interface
{ 
  /** Redefine String class to Accept numeric inference. */
  class NumString(str: String) 
  {
    /** Analyze if input is valid. */
    def isValid(low: Int, up: Int): Boolean =
    {
      ((!str.isBlank) && (str.matches("^\\d+$")) && (str.toInt >= low) && (str.toInt <= up))
    }
  }
  
  /** Implicit transformator of the String class. */
  implicit def numString(string: String) = new NumString(string)

  /**
   *  Prints welcome screen.
   */
  def printWelcome() 
  {
    println;
    
    print("______/\\\\\\____________/\\\\\\\\\\_____/\\\\\\\\\\\\\\\\\\\\______/\\\\\\\\\\\\\\\\\\_______________/\\\\\\____                \n".yellow.bold.blink)
    print(" __/\\\\\\\\\\\\\\________/\\\\\\\\////____/\\\\\\///////\\\\\\___/\\\\\\///////\\\\\\___________/\\\\\\\\\\____                 \n".yellow.bold.blink)       
    print("  _\\/////\\\\\\_____/\\\\\\///________\\///______/\\\\\\___\\/\\\\\\_____\\/\\\\\\_________/\\\\\\/\\\\\\____                   \n".yellow.bold.blink)      
    print("   _____\\/\\\\\\___/\\\\\\\\\\\\\\\\\\\\\\____________/\\\\\\//____\\///\\\\\\\\\\\\\\\\\\/________/\\\\/\\/\\\\\\_____         \n".yellow.bold.blink)     
    print("    _____\\/\\\\\\__/\\\\\\\\///////\\\\\\_________\\////\\\\\\____/\\\\\\///////\\\\\\_____/\\\\\\/__\\/\\\\\\____              \n".yellow.bold.blink)    
    print("     _____\\/\\\\\\_\\/\\\\\\______\\//\\\\\\___________\\//\\\\\\__/\\\\\\______\\//\\\\\\__/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_  \n".yellow.bold.blink)  
    print("      _____\\/\\\\\\_\\//\\\\\\______/\\\\\\___/\\\\\\______/\\\\\\__\\//\\\\\\______/\\\\\\__\\///////////\\\\\\//__            \n".yellow.bold.blink)  
    print("       _____\\/\\\\\\__\\///\\\\\\\\\\\\\\\\\\/___\\///\\\\\\\\\\\\\\\\\\/____\\///\\\\\\\\\\\\\\\\\\/_____________\\/\\\\\\____ \n".yellow.bold.blink)
    print("        _____\\///_____\\/////////_______\\/////////________\\/////////_______________\\///_____                                 \n".yellow.bold.blink)
 
    println;
    println("\t\t\tBY PABLO ACEREDA GARCIA & DAVID E. CRACIUNESCU".white.bold)
    println;
  }

  /**
   *  Prints option screen and returns chosen action.
   *  
   *  Return code:
   *  [1] Play game.
   *  [2] Change difficulty.
   *  [3] Quit game.
   *
   *  @return The code of the action.
   */
  def pickAction(): Int = 
  {
    println;
    println("Choose an action:"     .yellow.bold)
    println("[1] Play"              .cyan.bold)
    println("[2] Change difficulty" .cyan.bold)
    println("[3] Quit game"         .cyan.bold)
    println;
 
    val action: String = scala.io.StdIn.readLine();

    if (action.isValid(1, 3)) action.toInt
    else pickAction
  }

  /**
   *  Exits the game.
   */ 
  def exitGame(): Unit = {System.exit(0)} 

  /**
   *  Prints game difficulty description and asks user for choice.
   *  
   *  The difficulty levels are the following:
   *
   *  Level | Size Board |  Initial Seeds {vals} |  Movement {vals}
   *  -------------------------------------------------------------
   *    1   |    4x4     |      2 {2}            |    +1 {2}
   *    2   |    9x9     |      4 {2, 4}         |    +3 {2, 4}
   *    3   |   14x14    |      6 {2, 4, 8}      |    +5 {2, 4, 8}
   *    4   |   17x17    |      6 {2, 4, 8}      |    +6 {2, 4, 8}
   */ 
  def pickDifficulty(): Int = 
  { 
    println;
    println("Choose a difficulty:".white.bold)
    println("[1]. Easy            (4x4)".cyan.bold)
    println("[2]. Moderate        (9x9)".green.bold)
    println("[3]. Hard            (14x14)".yellow.bold)
    println("[4]. Extremely Hard  (17x17)".red.bold)
    println;
    
    val diff: String = scala.io.StdIn.readLine(); 
    
    if (diff.isValid(1, 4)) diff.toInt
    else pickDifficulty
  }

  /**
   *  Prints the current number of lives on screen with the given format. 
   */ 
  def printLives(max: Int, lives: Int): Unit =
  {
    printLivesAux(max, lives, 0)
  }

  /**
   *  Prints an individual heart-life on screen with the given format.
   */
  def printLivesAux(max: Int, lives: Int, current: Int): Unit =
  {
    if (current != max)
    {
      if      (current < lives)   print("<3".red.bold)
      else if (current == lives)  print("<3".red.bold.blink)
      else                        print("<3".black.bold)

      printLivesAux(max, lives, current + 1)
    }
    else println;
  }

  /**
   *  Prints the movement keys.
   */ 
  def printControls(): Unit =
  {
    println("Controls:".yellow.bold)       
    println;
    println("        ___          ".yellow.bold)               
    println("       | W |         ".yellow.bold)         
    println(" ___    ___    ___   ".yellow.bold)    
    println("| A |  | S |  | D |  ".yellow.bold)

  }

  /**
   *  Prints a two-dimensional square board array on screen recursively.
   *  This method triggers the printing for the whole board.
   *
   *  @param board The board to be printed on screen.
   */ 
  def printBoard(board: Array[Int]): Unit = printBoardAux(board, 0)

  /**
   *  Prints an element of a two-dimensional square board array on screen
   *  recursively.
   *
   *  @param board The board to be printed on screen.
   */
  private def printBoardAux(board: Array[Int], pos: Int): Unit =
  {
    print("\t")
    print(s"${board(pos)}".colorVal)
    
    if (pos == board.length - 1) println
    else
    {
      if ((pos + 1) % math.sqrt(board.size) == 0) println
      printBoardAux(board, pos + 1)
    }
  } 

  /**
   *  Prints end-screen and points.
   */
  def printEndScreen(points: Int): Unit =
  {
    println;
    
    print("\t\t _____  ___ ___  ________   _____ _   _ ___________          \n".red.bold.blink)
    print("\t\t|  __ \\/ _ \\|  \\/  |  ___| |  _  | | | |  ___| ___ \\     \n".red.bold.blink)
    print("\t\t| |  \\/ /_\\ \\ .  . | |__   | | | | | | | |__ | |_/ /      \n".red.bold.blink)
    print("\t\t| | __|  _  | |\\/| |  __|  | | | | | | |  __||    /         \n".red.bold.blink)
    print("\t\t| |_\\ \\ | | | |  | | |___  \\ \\_/ | \\_/ / |___| |\\ \\   \n".red.bold.blink)
    print("\t\t \\____|_| |_|_|  |_|____/   \\___/ \\___/\\____/\\_| \\_|   \n".red.bold.blink)
    println;                                                                     
 
    println(f"\t\tTotal points:${points}".yellow.bold)
  }
}
