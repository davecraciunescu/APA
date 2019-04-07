object Grid
{
  def empty(size: Int): Array[Int] = Array.ofDim[Int](size*size)

  /**
   *  Transforms the board stream into Arrays to operate with the values.
   */
  def toArray(grid: Stream[Int], dir: Direction = Left): Array[Int] =
  {
    val size: Int = grid.size
    val flat_stream: Array[Int] = grid.flatten.to[Array]

    dir match
    {
      case Left => Array.tabulate(size * size)((x: Int, y: Int) => flat_stream(x * size + y))
    }
  }

  /**
   *  Transforms the board array into Stream to avoid waster memory.
   */ 
  def toStream(grid: Array[Int], dir: Direction = Left): Stream[Int] =
  {
    val size: Int = grid.size

    def helper() = 
    {

      dir match
      {
        case Left => Stream.from(0).map(y => Stream.from(0).map(x => grid(y)(x)))
      }
    }
    helper().map(_.take(size)).take(size)
  }
}
