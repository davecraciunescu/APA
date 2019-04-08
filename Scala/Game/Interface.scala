import Color._
import scala.io.StdIn.{readLine, readInt}

/**
 *  Graphical interface and I/O interaction with user.
 *
 *      @author: Dave E. Craciunescu. Pablo Acereda Garcia
 *        @date: 2019.04.08
 *
 *        @todo: Add game board format.
 *               Add difficulty system.
 *               Add user action choice.
 *
 *   @changelog:
 *    -- 2019.04.08 -- Dave E.
 *      -- Add welcome screen ASCII Art.
 *      -- Add color coding.
 *
 *   @knownBugs:
 */
object Interface
{ 
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
 
    scala.io.StdIn.readInt()
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
  def piclDifficulty(): Int = 
  {
    
  }
  
  /**
   *  Prints the number of lives on screen.
   */ 
  def printLives(): Unit =
  {

  }

  /**
   *  Prints the movement keys.
   */ 
  def printKeyBindings(): Unit =
  {

  }

  /**
   *  Prints end-screen information and points.
   */
  def printEndScreen(points: Int): Unit =
  {
    println;
    print(" _____  ___ ___  ________   _____ _   _ ___________          \n".red.bold.blink)
    print("|  __ \\/ _ \\|  \\/  |  ___| |  _  | | | |  ___| ___ \\     \n".red.bold.blink)
    print("| |  \\/ /_\\ \\ .  . | |__   | | | | | | | |__ | |_/ /      \n".red.bold.blink)
    print("| | __|  _  | |\\/| |  __|  | | | | | | |  __||    /         \n".red.bold.blink)
    print("| |_\\ \\ | | | |  | | |___  \\ \\_/ | \\_/ / |___| |\\ \\   \n".red.bold.blink)
    print(" \\____|_| |_|_|  |_|____/   \\___/ \\___/\\____/\\_| \\_|   \n".red.bold.blink)
    println;                                                                     
  }

  def main(args: Array[String]): Unit =
  {
    welcome();
    val test: Int = pickAction();
  }
}
