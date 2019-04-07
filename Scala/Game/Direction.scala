/**
 *  Directions for the game are limited.
 *
 *  @note PABLO: Sealed Traits are the scala equivalent of java enums, but they
 *  can only be extended in the same file. Delete this note after reading it.
 */
sealed trait Direction

case object Left  extends Direction
case object Right extends Direction
case object Up    extends Direction
case object Down  extends Direction

/**
 *  Direction system the game will follow.
 *
 *     @author: Dave E. Craciunescu, Pablo Acereda Garcia
 *       @date: 2019.04.07
 *
 *       @todo: Figure out how to make this work normally.
 *              Delete the note that was left for Pablo.
 *
 *  @changelog:
 *    -- 2019.04.07 -- Dave E.
 *      -- Define available directions.
 *      -- Create reversal mechanism.
 *      -- Implement "nearest corner" movement.
 */
object Direction
{
  def nextL(dir: Direction): Direction =
  {
    dir match
    {
      case Left  => Down
      case Up    => Left
      case Right => Up
      case Down  => Right
    }
  }

  def nextR(dir: Direction): Direction =
  {
    dir match
    {
      case Left  => Up
      case Up    => Right
      case Right => Down
      case Down  => Left
    }
  }

  def opposite(dir: Direction): Direction = Direction.nextR(Direction.nextR(dir))
}
