object Test
{
  def testArray(size: Int): Array[Int] = (0 to size*size-1).toArray

  def empty(size: Int): Array[Int] = Array.ofDim[Int](size * size)

  /**
     *  Prints the two-dimensional board array on screen recursively.
  */
  def printBoardAux(board: Array[Int], pos: Int): Unit =
  {
    if (pos != board.length - 1)
    {
      if (pos == 0) print(s"\t${board(pos)}")
      
      else if ((pos + 1) % math.sqrt(board.size) == 0) println(s"\t${board(pos)}")

      else print(s"\t${board(pos)}")

      printBoardAux(board, pos + 1)
    }
    else println(s"\t${board(pos)}")
  }

  def printBoard(board: Array[Int]): Unit = 
  {
    print(" ")
    printBoardAux(board, 0)
  }
  
  def main(args: Array[String]): Unit =
  {
    val test1 = testArray(4)
    val test2 = empty(4)
  
    printBoard(test1);
  }

}

