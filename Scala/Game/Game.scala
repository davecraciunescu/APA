/**
 * Main code of the program
 */
object Game
{
  def main (args: Array[String])
  {
    // Title
    Interface.printWelcome()
    /*
     * Actions:
     * [1] Play Game.
     * [2] Change Difficulty.
     * [3] Quit Game.
     */
    val x: Int = Interface.pickAction() 
    x match 
    {
      case 1 => println("Being developed")
      case 2 => Interface.pickDifficulty()
      case 3 => Interface.exitGame()
    }

  }
}
