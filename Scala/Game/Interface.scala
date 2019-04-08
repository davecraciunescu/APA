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
  def welcome() 
  {
    println;
    print("______/\\\\\\____________/\\\\\\\\\\_____/\\\\\\\\\\\\\\\\\\\\______/\\\\\\\\\\\\\\\\\\_______________/\\\\\\____                \n".red.bold.blink)
    print(" __/\\\\\\\\\\\\\\________/\\\\\\\\////____/\\\\\\///////\\\\\\___/\\\\\\///////\\\\\\___________/\\\\\\\\\\____                 \n".red.bold.blink)       
    print("  _\\/////\\\\\\_____/\\\\\\///________\\///______/\\\\\\___\\/\\\\\\_____\\/\\\\\\_________/\\\\\\/\\\\\\____                   \n".red.bold.blink)      
    print("   _____\\/\\\\\\___/\\\\\\\\\\\\\\\\\\\\\\____________/\\\\\\//____\\///\\\\\\\\\\\\\\\\\\/________/\\\\/\\/\\\\\\_____         \n".red.bold.blink)     
    print("    _____\\/\\\\\\__/\\\\\\\\///////\\\\\\_________\\////\\\\\\____/\\\\\\///////\\\\\\_____/\\\\\\/__\\/\\\\\\____              \n".red.bold.blink)    
    print("     _____\\/\\\\\\_\\/\\\\\\______\\//\\\\\\___________\\//\\\\\\__/\\\\\\______\\//\\\\\\__/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_  \n".red.bold.blink)  
    print("      _____\\/\\\\\\_\\//\\\\\\______/\\\\\\___/\\\\\\______/\\\\\\__\\//\\\\\\______/\\\\\\__\\///////////\\\\\\//__            \n".red.bold.blink)  
    print("       _____\\/\\\\\\__\\///\\\\\\\\\\\\\\\\\\/___\\///\\\\\\\\\\\\\\\\\\/____\\///\\\\\\\\\\\\\\\\\\/_____________\\/\\\\\\____ \n".red.bold.blink)
    print("        _____\\///_____\\/////////_______\\/////////________\\/////////_______________\\///_____                                 \n".red.bold.blink)
 
    println;
    println("\t\t\tBY PABLO ACEREDA GARCIA & DAVID E. CRACIUNESCU".yellow.bold)
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
    println("Choose an action:".red.bold)
    println("[1] Play".cyan.bold)
    println("[2] Change difficulty (Default = 1)".cyan.bold)
    println("[3] Quit game".cyan.bold)
    println;
 
    scala.io.StdIn.readInt()
  }

  /**
   *  Exits the game.
   */ 
  def exitGame() {}

  /**
   *  Changes game difficulty.
   */ 
  def difficulty() {}

  def main(args: Array[String]): Unit =
  {
    welcome();
    val test: Int = pickAction();
  }
}
