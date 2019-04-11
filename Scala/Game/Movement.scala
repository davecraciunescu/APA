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
  private def moveHor(board: List[Int], col: Int): List[Int] =
  {
    val  m1 = moveRight(board, col)
    val sum = merge(m1, col, 0)
    val  m2 = moveRight(sum, col)
    m2
  }

  /**
   * Movement structure is generalized to a single method. All the movements are
   * based on the right movement.
   *
   * @param board List which contains the tiles.
   * @param col   Number of columns.
   *
   * @return Board having moved. 
   */
  private def moveVer(board: List[Int], col: Int): List[Int] =
  {
    val  m1 = moveDown(board, col)
    val sum = merge(m1, col, 0)
    val  m2 = moveDown(sum, col)
    m2
  }

  /**
   * Returns the tiles of a list moved to the right. 
   * Tiles with a 0 count as empty.
   *
   * @param board The List which holds the tiles.
   * @param col   Number of columns. 
   *
   * @return Move the board to the right.
   */
  private def moveRight (board: List[Int], col: Int): List[Int] = 
  {
    if (board != Nil) 
      moveRightAux(getN(col,board), col, 0):::moveRight(remove(col, board), col);
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
       board.head :: moveRightAux (board.tail, col, pos + 1)
     }
     else if(board.head != 0 )
     {
       if(board.tail.head == 0)
       {
         0 :: moveRightAux (board.head :: board.tail.tail, col, pos + 1)
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
  def moveLeft(board: List[Int], col: Int): List[Int] =
  {
    // The method is actually the movement to the right of the reversed
    // matrix. 
    // The matrix again reversed to obtain the original matrix.
    moveHor(board.reverse, col).reverse
  }

  /**
   * Moves the tiles of a matrix down.
   *
   * @param board List which contains the tiles.
   * @param col   Number of columns.
   *
   * @return Board moved down.
   */
  def moveDown(board: List[Int], col: Int): List[Int] = 
  {
    if (board.isEmpty) Nil
    else if (board.head > 0 && col < board.length && get(col, board) == 0)
    {
      val list = place(board.head, col+1, board)
      0 :: moveDown(list.tail, col)
    } else
    {
      board.head :: moveDown(board.tail, col)
    }
  }

  /**
   * Moves the tiles of a matrix up.
   *
   * @param board List which contains the tiles.
   * @param col   Number of columns.
   *
   * @return Board moved up.
   */
  def moveUp(board: List[Int], col: Int): List[Int] =
  {
    // It generates the reverse matrix in order to move the tiles.
    // The reverse of the list obtained by that method is the original matrix.
    moveVer(board.reverse, col).reverse
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
   *  Retrieves the specified element.
   *
   *  @param n     Position.
   *  @param board List which contains the tiles.
   *
   *  @return Element n.
   */
  def get(n: Int, board: List[Int]): Int=
  {
    if (n == 0) board.head
    else get(n - 1, board.tail)
  }
 
 
  /**
   *  Places a value in a List in the given position.
   *
   *  The method searches recursively for the given position and appends back to
   *  itself the values that to not match with the position.
   *
   *  @param board The List to be placed in.
   */
  def place(num: Int, pos: Int, board: List[Int]): List[Int] =
  {
    if      (board.length == 0) Nil
    else if (pos == 1) num :: board.tail
    else    board.head :: place(num, (pos - 1), board.tail)
  }

  /**
   *  Choose a movement according to the specified string.
   */
  def move(movement: String, board: List[Int], col: Int): List[Int] =
  {
    movement match
    {
      case ("s"|"S") => moveVer(board, col)
      case ("a"|"A") => moveLeft(board, col)
      case ("d"|"D") =>  moveHor(board, col)
      case ("w"|"W") =>   moveUp(board, col, col * col)
    }
  }
}
