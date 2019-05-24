import scala.util.Random

/**
 *  Represents the board where the game will be played.
 *
 *     @author: Dave E. Craciunescu. Pablo Acereda Garcia
 *       @date: 2019.04.06
 * 
 *       @todo: Create movement visualization.
 *
 *  @changelog:
 *    -- 2019.04.06 -- Dave E.
 *      -- Add empty initialization mechanism.
 *      -- Implement multiple size board visualization.
 *    
 *    -- 2019.04.10 -- Dave E.
 *      -- Migrate multiple size board visualization to Interface.
 *      -- Implement random and difficulty-based seeding.
 *    
 *    -- 2019.04.11 -- Dave E.
 *      -- Create color coding for values.
 *      -- Add not-random initialization mechanism.
 *
 *  @knownBugs:
 *      
 */
object Board
{
  /**
   *  Returns a new empty two-dimensional square board list of size 'size'
   *  
   *  @param size The size of the list.
   */ 
  def initBoard(size: Int): List[Int] = List.fill(size*size)(0)

  /**
   *  Returns if a list is initialized but not seeded.
   */
  def isEmpty(board: List[Int]): Boolean = (board.max == 0) 

  /**
   *  Returns the value of a List element in a given position.
   */
  private def getValue(pos: Int, board: List[Int]): Int = board(pos)
  
  /**
   *  Places a value in a List in the given position.
   *  
   *  The method searches recursively for the given position and appends back to
   *  itself the values that do not match with the position.
   *
   *  @param board The List to be placed in.
   */ 
  private def placeValue(num: Int, pos: Int, board: List[Int]): List[Int] =
  {
    if      (board.length == 0) Nil
    else if (pos == 0) num :: board.tail
    else    board.head :: placeValue(num, (pos - 1), board.tail)
  }

  /**
   *  Seeds a board with the given values.
   *  
   *  @param board: The list to be seeded.
   *  @param level: The level of difficulty of the game.
   *
   *  @return Reset board if can't seed.
   */
  private def addSeeds(board: List[Int], seeds: List[Int]): List[Int] =
  {
    if      (board.length == 0) Nil
    else if (  seeds.size == 0) board
    else if (getFreeSpots(board).size < seeds.size) { initBoard(board.length) }
    else
    {
      val  rand:       Int = scala.util.Random.nextInt(seeds.size)
      val added: List[Int] = placeValue(seeds(0), getFreeSpots(board)(rand), board)
      addSeeds(added, seeds.tail)
    }
  }

  /**
   *  Returns a list with the indexes of all free spots.
   */
  private def getFreeSpots(board: List[Int]): List[Int] = 
  {
    Random.shuffle(board.zipWithIndex.collect { case (0, i) => i }) 
  } 

  /**
   *  Returns a list with a random selection from the provided values.
   *
   *  @param amount The size of the created list.
   *  @param values The list of values to select from.
   */
  private def genSeeds(amount: Int, values: List[Int]): List[Int] =
  {
    List.fill(amount)(values(scala.util.Random.nextInt(values.length)))
  }

  /**
   *  Function to separate Initial Seeding from movement seeding.
   *
   *  The difficulty levels will generate the following seeds:
   *  Level | Size Board |  Initial Seeds {vals} |  Movement {vals}
   *  -------------------------------------------------------------
   *    1   |    4x4     |      2 {2}            |    +1 {2}
   *    2   |    9x9     |      4 {2, 4}         |    +3 {2, 4}
   *    3   |   14x14    |      6 {2, 4, 8}      |    +5 {2, 4, 8}
   *    4   |   17x17    |      6 {2, 4, 8}      |    +6 {2, 4, 8}
   *
   *  @param board  The board to seed with the values.
   *  
   *  @knownBugs: Init in empty board always 
   */
  def seedBoard(board: List[Int]): List[Int] =
  {
    if (isEmpty(board)) initSeed(board)
    else moveSeed(board)
  }

  /** 
   *  Returns a board with initial seeding according to difficulty.
   *  
   *  @param board The board to be seeded.
   */ 
  private def initSeed(board: List[Int]): List[Int] =
  { 
    // Pass the sizes of the boards instead the difficulty as parameter.
    numElems(board) match
    {
      case 16  => addSeeds(board, genSeeds(2, List(2)))
      case 81  => addSeeds(board, genSeeds(4, List(2, 4)))
      case 196 => addSeeds(board, genSeeds(6, List(2, 4, 8)))
      case 289 => addSeeds(board, genSeeds(6, List(2, 4, 8)))
    }
  }

  /** 
   *  Returns a board with movement seeding according to difficulty.
   *  
   *  @param board The board to be seeded.
   */ 
  private def moveSeed(board: List[Int]): List[Int] =
  {
    numElems(board) match
    {
      case 16  => addSeeds(board, genSeeds(1, List(2)))
      case 81  => addSeeds(board, genSeeds(3, List(2, 4)))
      case 196 => addSeeds(board, genSeeds(5, List(2, 4, 8)))
      case 289 => addSeeds(board, genSeeds(6, List(2, 4, 8)))
    }
  }
  
  /**
   *  Returns the points of the game. This will amount to the sum of the elems.
   *
   *  @param board The board itself.
   */ 
  def getPoints(board: List[Int]): Int = sum(board)
  
  /**
   *  Returns the number of elements in a list.
   */ 
  def numElems(l: List[Int]): Int = 
  {
    if(l == Nil) 0
    else 1 + numElems(l.tail)
  }
 
  /**
   *  Adds two elements together.
   */
  private def add(x: Int, y: Int): Int = 
  {  
    if(y == 0) x
    else add(x, y - 1) + 1
  }

  /**
   * Sums the values of a whole list.
   */
  def sum(x: List[Int]): Int =
  {
    if (numElems(x) == 0) 0
    else add(x.head, sum(x.tail))
  }

}
