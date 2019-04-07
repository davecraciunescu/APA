/**
 * Set of available Directions for the 2048 Game.  
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
 *              This works, I promise. I just have to get it working.
 *
 *  @changelog:
 *    -- 2019.04.07 -- Dave E.
 *      -- Define available directions.
 *      -- Create reversal mechanism.
 *      -- Implement "nearest corner" movement.
 */
object Direction
{
  /**
   *  Generates the next direction aiming at the left corner.
   */
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

  /**
   *  Generates the next direction aiming at the right corner.
   */
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
  
  /**
   *  Acts in the opposite direction of the provided one.
   *  This method is used to UNDO a movement.
   */
  def opposite(dir: Direction): Direction = Direction.nextR(Direction.nextR(dir))
}
