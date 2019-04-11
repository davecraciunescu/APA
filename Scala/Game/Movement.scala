/**
 *  Definition of the movements in the 2048 game.
 *  
 *      @author: Dave E. Craciunescu. Pablo Acereda Garcia.
 *        @date: 2019.04.11
 *  
 *        @todo: Make work for up and down movements.
 *
 *   @changelog:
 *    -- 2019.04.08 -- Pablo A.
 *      -- Create movement implementation.
 *      -- Implement cell joining.
 *
 *   @knownBugs:
 *    -- Does not operate properly for up and down movements.
 */ 
object Movement
{
  /**
   * Movement structure is generalized to a single method. All the movements are
   * based on the right movement.
   *
   * @param board List which contains the tiles.
   * @param col   Number of columns.
   *
   * @return Board having moved. 
   */
   private def moveGen(board: List[Int], col: Int, pos: Int): List[Int] =
   {
     val  m1 = moveRight(board, col, 0)
     val sum = merge(m1, col, 0)
     val  m2 = moveRight(sum, col, 0)
     m2
   }
 
  /**
   * Returns the tiles of a list moved to the right. 
   * Tiles with a 0 count as empty.
   *
   * @param board The List which holds the tiles.
   * @param col   Number of columns. 
   * @param pos   Position to iterate whithin the board.
   *
   * @return Move the board to the right.
   */
   private def moveRight (board: List[Int], col: Int, pos: Int): List[Int] = 
   {
     if (board != Nil) 
       moveRightAux(getN(col,board), col, pos):::moveRight(remove(col, board), col, pos);
     else board
   }
  
  /**
   * Iterates through the list and moves the tiles to the right.
   *
   * @param board The List which holds the tiles.
   * @param col   Number of columns. 
   * @param pos   Position to iterate whithin the board.
   *
   * @return Element moved to the right.
   */
   private def moveRightAux(board: List[Int], col: Int, pos: Int): List[Int] =
   {
     if(board.tail == Nil) board
     else if((pos + 1 ) % col == 0)
     {
       board.head :: moveRight (board.tail, col, pos + 1)
     }
     else if(board.head != 0 )
     {
       if(board.tail.head == 0)
       {
         0 :: moveRightAux(board.head :: board.tail.tail, col, pos + 1)
       }
       else
       {
         if(board.reverse.head == 0) 
           0 :: moveRightAux((board.reverse.tail).reverse, col, pos + 1)
         else board.head :: moveRightAux(board.tail, col, pos + 1)
       } 
     }
     else
     {
       board.head :: moveRightAux(board.tail, col, pos + 1)
     }
   }

  /**
   * Merges contiguos tiles if they have the same value. 
   * Tiles once merged will not be remerged again.
   *
   * @param board The List which holds the tiles.
   * @param col   The number of columns. 
   * @param pos   Position to iterate whithin the board.
   *
   * @return Tiles merged.
   */
  def merge(board: List[Int], col: Int, pos: Int): List[Int] =
  {
    if(board.tail == Nil) board
    else if((pos + 1) % col == 0)
    {
     board.head :: merge(board.tail, col, pos+1)
    }
    else
    {
      if(board.head != 0 && board.tail.head == board.head)
      {
        val sum = board.head * 2
        val tab = sum :: board.tail.tail
       
        if(tab.tail == Nil) 0 :: tab
        else
        {
          if(tab.head == tab.tail.head) 0 :: tab.head :: merge(tab.tail, col, pos + 2)
          else 0 :: merge(tab, col, pos+1)
        }
      }
    else board.head :: merge(board.tail, col, pos + 1)
    }
  }
  
  /**
   * Moves the tiles of a matrix to the left.
   *
   * @param board The List which holds the tiles.
   * @param col   Number of columns.
   *
   * @return Board moved to the left.
   */
  def moveLeft(board: List[Int], col: Int, pos: Int): List[Int] =
  {
    // The method is actually the movement to the right of the reversed
    // matrix. 
    // The matrix again reversed to obtain the original matrix.
    moveGen(board.reverse, col, 0).reverse
  }

  /**
   * Moves the tiles of a matrix down.
   *
   * @param board List which contains the tiles.
   * @param col   Number of columns.
   * @param size  Size of the board.
   * @param pos   Position to iterate whithin the board.
   *
   * @return Board moved down.
   */
  def moveDown(board: List[Int], col: Int, size: Int, pos: Int): List[Int] = 
  {
    // It uses a transposed matrix to fill move the tiles.
    // After the final matrix is obtained, it is then again transposed to obtain
    // the original matrix.
    val tras = trans(board, col, size, pos)
    val mov  = moveRight(tras, col, pos)

    trans(moveGen(trans(board, col, size, pos), col, pos), col, size, pos)
  }

  /**
   * Moves the tiles of a matrix up.
   *
   * @param board List which contains the tiles.
   * @param col   Number of columns.
   * @param size  Size of the board.
   * @param pos   Position to iterate within the method.
   *
   * @return Board moved up.
   */
  def moveUp(board: List[Int], col: Int, size: Int, pos: Int): List[Int] =
  {
    // It generates the transposed matrix and sends it to the left movement.
    // The transposed of the list obtained by that method is the original
    // matrix.
    val tras = trans(board, col, size, pos)
    val mov = moveLeft(tras, col, pos)
   
    trans(moveLeft(trans(board, col, size, pos), col, pos), col, size, pos)
  }

  /**
   * Creates the transposed of a given matrix.
   *
   * @param board List which contains the tiles.
   * @param col   Number of columns.
   * @param size  Size of the board.
   * @param pos   Position to iterate within the method.
   *
   * @return The transpose of the matrix.
   */
  def trans (board: List[Int], col: Int, size: Int, pos: Int): List[Int] = 
  {
    if (board == Nil) board
    else
    {
      if (size == col) Nil
      else if (pos >= size) trans(board.tail, col, size - 1, 0)
      else get(pos, board) :: trans(board, col, size, pos + col)
    }
  }
 
  /**
   * Retrieves the first n elements from a list.
   *
   * @param n Position.
   * @param l List which contains the tiles. 
   *
   * @return First n elements.
   */
  def getN(n: Int, l: List[Int]): List[Int] =
  {
    if (n == 0) Nil
    else l.head :: getN(n - 1, l.tail)
  }
 
  /**
   * Removes the first n elements from a list.
   *
   * @param n Position.
   * @param l List which contains the tiles.
   *
   * @return List without first n elements.
   */
  def remove(n: Int, l: List[Int]): List[Int] =
  {
    if (l == Nil) Nil
    else if (n == 0) l
    else remove(n - 1, l.tail)
  }
 
  /**
   * Retrieves the specified element.
   *
   * @param n     Position.
   * @param board List which contains the tiles.
   *
   * @return Element n.
   */
  def get(n: Int, board: List[Int]): Int=
  {
    if (n == 0) board.head
    else get(n - 1, board.tail)
  }
 
  def move (movement: String, board: List[Int], col: Int): List[Int] =
  {
    movement match
    {
      case ("s"|"S") => moveDown(board, col, col * col, 0)
      case ("a"|"A") => moveLeft(board, col,            0)
      case ("d"|"D") =>  moveGen(board, col,            0)
      case ("w"|"W") =>   moveUp(board, col, col * col, 0)
    }
  }
}
