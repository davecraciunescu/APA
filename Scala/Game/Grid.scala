import scala.util.Random

/**
 *  Represents the grid where the game will be played.
 *
 *     @author: Dave E. Craciunescu. Pablo Acereda Garcia
 *       @date: 2019.04.06
 * 
 *       @todo: Add not-random initialization mechanism.
 *              Create movement visualization.
 *              Create color coding for values.
 *
 *  @changelog:
 *    -- 2019.04.06 -- Dave E.
 *      -- Add empty initialization mechanism.
 *      -- Implement multiple size board visualization.
 *    -- 2019.04.10 -- Dave E.
 *      -- Migrate multiple size board visualization to Interface.
 *      -- Implement random and difficulty-based seeding.
 *
 *  @knownBugs:
 *      
 */
object Grid
{

  /**
   *  Returns a new empty two-dimensional square grid list of size 'size'
   *  
   *  @param size The size of the list.
  */ 
  def init(size: Int): List[Int] = List.fill(size*size)(0)

  /**
   *  Places a value in a List in the given position.
   *  
   *  The method searches recursively for the given position and appends back to
   *  itself the methods that do not match with the position.
   *
   *  @param grid The List to be placed in.
   */ 
  def placeValue(num: Int, pos: Int, grid: List[Int]): List[Int] =
  {
    if (grid.length == 0) Nil
    else if (pos == 0) num :: grid.tail
    else grid.head :: placeValue(num, (pos - 1), grid.tail)
  }

  /**
   *  Returns the value of a List element in a given position.
   */
  def getValue(pos: Int, grid: List[Int]): Int = grid(pos)

  /**
   *  Returns a list with the indexes of all free spots.
   */
  def getFreeSpots(grid: List[Int]): List[Int] = 
  {
    grid.zipWithIndex.collect { case (0,i) => i } 
  }
  
  /**
   *  Initializes the board with seeds depending on difficulty.
   *
   *  The difficulty levels will generate the following seeds:
   *  Level | Size Board |  Initial Seeds {vals} |  Movement {vals}
   *  -------------------------------------------------------------
   *    1   |    4x4     |      2 {2}            |    +1 {2}
   *    2   |    9x9     |      4 {2, 4}         |    +3 {2, 4}
   *    3   |   14x14    |      6 {2, 4, 8}      |    +5 {2, 4, 8}
   *    4   |   17x17    |      6 {2, 4, 8}      |    +6 {2, 4, 8}
   *  
   *  @param board: The list to be seeded.
   *  @param level: The level of difficulty of the game.
   */
  def seedBoard(board: List[Int], seeds: List[Int]): List[Int]
  {
    
    // if list is null
    // else
    //  iterate over list
    //    if current val == 0
    //      if there are more spots
    //        random generator decides
    //      if there are exact spots
    //        fill them all
    //      else
    //        can't seed
    //    else
  }
  
  /**
   *  Returns a list with a random selection from the provided values.
   *
   *  @param amount The size of the created list.
   *  @param values The list of values to select from.
   */
  def genSeeds(amount: Int, values: List[Int]): List[Int] =
  {
    List.fill(amount)(values(scala.util.Random.nextInt(values.length)))
  }

  def main (args: Array[String]): Unit =
  {
    val values = List(2, 4, 8)
    genSeed(6, values).foreach(println)
  }
}
