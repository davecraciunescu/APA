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
   *  Returns a new empty two-dimensional square grid array of size 'size'
   *  
   *  @param size The size of the array.
   */
  def init(size: Int): Array[Int] = Array.ofDim[Int](size * size)

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
   *  @param board: The array to be seeded.
   *  @param level: The level of difficulty of the game.
   */
  def seed(board: Array[Int], level: Int)
  {
         
  }
}
