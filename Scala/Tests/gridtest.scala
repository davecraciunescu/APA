sealed trait Direction

case object Left extends Direction
case object Right extends Direction
case object Up extends Direction
case object Down extends Direction

object Direction
{
  def nextL(dir: Direction): Direction =
  {
    dir match
    {
      case Left   => Down
      case Up     => Left
      case Right  => Up
      case Down   => Right
    }
  }

  def nextR(dir: Direction): Direction =
  {
    dir match
    {
      case Left   => Up
      case Up     => Right
      case Right  => Down 
      case Down   => Left
    }
  }

  def opposite(dir: Direction): Direction = Direction.nextR(Direction.nextR(dir))
}

object Grid
{
  def empty(size: Int): Array[Array[Int]] = Array.ofDim[Int](size, size)

  def of(s: Stream[Stream[Int]], dir: Direction = Left): Array[Array[Int]] =
  {
    val size: Int = s.size
    val flat_stream: Array[Int] = s.flatten.to[Array]

    dir match
    {
      case Left => Array.tabulate(size, size)((x: Int, y: Int) => flat_stream(x * size + y))
      case Right => Array.tabulate(size, size)((x: Int, y: Int) => flat_stream((size - x - 1) * size + (size - 1)))
      case Up => Array.tabulate(size, size)((x: Int, y: Int) => flat_stream((size - y - 1) * size + x))
      case Down => Array.tabulate(size, size)((x: Int, y: Int) => flat_stream(y * size + (size - x - 1)))
    }
  }

  def toStreams(grid: Array[Array[Int]], dir: Direction = Left): Stream[Stream[Int]] =
  {
    val size: Int = grid.size
    def helper() =
    {
      dir match
      {
        case Left => Stream.from(0).map(y => Stream.from(0).map(x => grid(y)(x)))
        case Right => Stream.from(0).map(y => Stream.from(0).map(x => grid(size - y - 1)(size - x - 1)))
        case Up => Stream.from(0).map(y => Stream.from(0).map(x => grid(x)(size - y - 1)))
        case Down => Stream.from(0).map(y => Stream.from(0).map(x => grid(size - x - 1)(y)))
      }
    }
      helper().map(_.take(size)).take(size)
  }
}

object theMain
{ 
  def main(args: Array[String]): Unit =
  {
    val s = 
      (
          ( 1 ::  2 ::  3 ::  4 :: Nil).toStream
      ::  ( 5 ::  6 ::  7 ::  8 :: Nil).toStream
      ::  ( 9 :: 10 :: 11 :: 12 :: Nil).toStream
      ::  (13 :: 14 :: 15 :: 16 :: Nil).toStream
      ::  Nil
      ).toStream

    val test = Grid.of(s, Left)

    test.forEach(_.forEach(print(_)));
  }
}
