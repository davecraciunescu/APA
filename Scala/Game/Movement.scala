/**
 *  Movement of the different values inside the board.
 *
 *       @author: Dave E. Craciunescu, Pablo Acereda Garcia
 *         @date: 2019.04.07
 *
 *         @todo: Implement with single-index data structures.
 *                Make work with Stream data structure.
 *                Make work with rest of the program.
 *
 *    @changelog:
 *      -- 2019.04.07 -- Dave E.
 *        -- Define movement and basic actions.
 *        -- Create tile-merging mechanism.
 */ 
class Movement[A](val extract: A => Int, val merge: (A, A) => A, val nullTile: A)
{
  /**
   *  Merge successive tiles having same value together.
   *  
   *  @param s stream of tiles' values
   *  @return merged version of s
   */ 
  def merge_tiles(s: Seq[A]): Stream[A] =
  {
    s match
    {
      case Nil                                              => Stream.empty[A]
      case Seq(t)                                           => t #:: Stream.empty[A]
      case t1 #:: t2 #:: q  if (extract(t1) == extract(t2)) => merge(t1, t2) #:: merge_tiles(q)
      case t1 #:: t2 #:: q                                  => t1 #:: merge_tiles(t2 #:: q)
    }
  }

  /**
   *  Moves the values inside the board based on a Direction.
   *
   *  @param s Values on the board as a Stream.
   */ 
  def play_move(num: Int)(s: Stream[A]): Stream[A] =
    merge_tiles(s.filter(extract(_) != 0))
      .append(Stream.continually(nullTile))
      .take(num)  
}

object Movement
{
  def ofInt(): Movement[Int] = new Movement[Int](a => a, (a, b) => a + b, 0)
}
