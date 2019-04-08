/**
 *  String color coding mechanism.
 *
 *      @author: Dave E. Craciunescu. Pablo Acereda Garcia
 *        @date: 2019.04.08
 *
 *        @todo: 
 *
 *   @changelog:
 *    -- 2019.04.08 -- Dave E.
 *      -- Add basic color support.
 *      -- Add background color support.
 *      -- Add support for special features.
 *
 *   @knownBugs:
 *          
 */
trait Color
{
  // Needed to make implicit def work.
  import scala.language.implicitConversions
  
  implicit def hasColor(s: String) = new ColorString(s)

  /**
   *  Different color methods.
   */
  class ColorString(s: String)
  {
    import Console._

    // Basic colors.
    
    /** Prints the string in ANSI BLACK.        */
    def      black = BLACK      + s + RESET
    
    /** Prints the string in ANSI RED.          */
    def        red = RED        + s + RESET
    
    /** Prints the string in ANSI GREEN.        */
    def      green = GREEN      + s + RESET
    
    /** Prints the string in ANSI YELLOW.       */
    def     yellow = YELLOW     + s + RESET
    
    /** Prints the string in ANSI BLUE.         */
    def       blue = BLUE       + s + RESET
    
    /** Prints the string in ANSI MAGENTA.      */
    def    magenta = MAGENTA    + s + RESET
    
    /** Prints the string in ANSI CYAN.         */
    def       cyan = CYAN       + s + RESET
    
    /** Prints the string in ANSI WHITE.        */
    def      white = WHITE      + s + RESET

    ///////////////////////////////////////////////////////////////////////////

    // Background colors.

    /** Prints the background in ANSI BLACK.    */
    def    onBlack = BLACK_B    + s + RESET
    
    /** Prints the background in ANSI RED.      */
    def      onRed = RED_B      + s + RESET
    
    /** Prints the background in ANSI GREEN.    */
    def    onGreen = GREEN_B    + s + RESET
    
    /** Prints the background in ANSI YELLOW.   */
    def   onYellow = YELLOW_B   + s + RESET
    
    /** Prints the background in ANSI BLUE.     */
    def     onBlue = BLUE_B     + s + RESET
    
    /** Prints the background in ANSI MAGENTA.  */
    def  onMagenta = MAGENTA_B  + s + RESET
    
    /** Prints the background in ANSI CYAN.     */
    def     onCyan = CYAN_B     + s + RESET
    
    /** Prints the background in ANSI WHITE.    */
    def    onWhite = WHITE_B    + s + RESET
  
    ///////////////////////////////////////////////////////////////////////////

    // Special text features.

    /** Prints the text in bold.                */
    def       bold = BOLD       + s + RESET
    
    /** Prints the text underlined.             */
    def underlined = UNDERLINED + s + RESET
    
    /** Prints the text blinking.               */
    def      blink = BLINK      + s + RESET
  }
}

object Color extends Color
