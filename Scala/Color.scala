import scala.language.implicitConversions

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
 *      -- Add string color formatting support.
 *    -- 2019.04.10 -- Dave E.
 *      -- Define color codes for 2048 Game. 
 *      -- Add bright color support.
 *
 *   @knownBugs:
 *    -- Color does not work in Windows terminal.
 *          
 */
trait Color
{
  implicit def hasColor(s: String) = new ColorString(s)

  /**
   *  Different color methods.
   */
  class ColorString(s: String)
  {
    import Console._ 
    
    // Custom foreground colors.

    /** Color code for Bright Black.    */
    final val B_BLACK     = "\u001B[90m"

    /** Color code for Bright Red.      */
    final val B_RED       = "\u001B[91m"

    /** Color code for Bright Green.    */
    final val B_GREEN     = "\u001B[92m" 

    /** Color code for Bright Yellow.   */
    final val B_YELLOW    = "\u001B[93m"

    /** Color code for Bright Blue.     */
    final val B_BLUE      = "\u001B[94m"

    /** Color code for Bright Magenta.  */
    final val B_MAGENTA   = "\u001B[95m"  

    /** Color code for Bright Cyan.     */
    final val B_CYAN      = "\u001B[96m"  

    /** Color code for Bright White.    */
    final val B_WHITE     = "\u001B[97m"  
    

    // Custom background colors.
    
    /** Color code for Bright Black.    */
    final val B_BLACK_B   = "\u001B[100m"

    /** Color code for Bright Red.      */
    final val B_RED_B     = "\u001B[101m"

    /** Color code for Bright Green.    */
    final val B_GREEN_B   = "\u001B[102m" 

    /** Color code for Bright Yellow.   */
    final val B_YELLOW_B  = "\u001B[103m"

    /** Color code for Bright Blue.     */
    final val B_BLUE_B    = "\u001B[104m"

    /** Color code for Bright Magenta.  */
    final val B_MAGENTA_B = "\u001B[105m"  

    /** Color code for Bright Cyan.     */
    final val B_CYAN_B    = "\u001B[106m"  

    /** Color code for Bright White.    */
    final val B_WHITE_B   = "\u001B[107m"  

    //////////////////////////////////////////////////////////////////////////

    // Predefined foreground colors.
    
    /** Prints the string in ANSI BLACK.              */
    def             black = BLACK       + s + RESET
    
    /** Prints the string in ANSI RED.                */
    def               red = RED         + s + RESET
    
    /** Prints the string in ANSI GREEN.              */
    def             green = GREEN       + s + RESET
    
    /** Prints the string in ANSI YELLOW.             */
    def            yellow = YELLOW      + s + RESET
    
    /** Prints the string in ANSI BLUE.               */
    def              blue = BLUE        + s + RESET
    
    /** Prints the string in ANSI MAGENTA.            */
    def           magenta = MAGENTA     + s + RESET
    
    /** Prints the string in ANSI CYAN.               */
    def              cyan = CYAN        + s + RESET
    
    /** Prints the string in ANSI WHITE.              */
    def             white = WHITE       + s + RESET

    // Custom foreground colors.

    /** Prints the string in ANSI BRIGHT BLACK.       */
    def       brightBlack = B_BLACK     + s + RESET
    
    /** Prints the string in ANSI BRIGHT RED.         */
    def         brightRed = B_RED       + s + RESET
    
    /** Prints the string in ANSI BRIGHT GREEN.       */
    def       brightGreen = B_GREEN     + s + RESET
    
    /** Prints the string in ANSI BRIGHT YELLOW.      */
    def      brightYellow = B_YELLOW    + s + RESET
    
    /** Prints the string in ANSI BRIGHT BLUE.        */
    def        brightBlue = B_BLUE      + s + RESET
    
    /** Prints the string in ANSI BRIGHT MAGENTA.     */
    def     brightMagenta = B_MAGENTA   + s + RESET
    
    /** Prints the string in ANSI BRIGHT CYAN.        */
    def        brightCyan = B_CYAN      + s + RESET
    
    /** Prints the string in ANSI BRIGHT WHITE.       */
    def       brightWhite = B_WHITE     + s + RESET
    
    ///////////////////////////////////////////////////////////////////////////

    // Predefined background colors.

    /** Prints the background in ANSI BLACK.          */
    def           onBlack = BLACK_B     + s + RESET
    
    /** Prints the background in ANSI RED.            */
    def             onRed = RED_B       + s + RESET
    
    /** Prints the background in ANSI GREEN.          */
    def           onGreen = GREEN_B     + s + RESET
    
    /** Prints the background in ANSI YELLOW.         */
    def          onYellow = YELLOW_B    + s + RESET
    
    /** Prints the background in ANSI BLUE.           */
    def            onBlue = BLUE_B      + s + RESET
    
    /** Prints the background in ANSI MAGENTA.        */
    def         onMagenta = MAGENTA_B   + s + RESET
    
    /** Prints the background in ANSI CYAN.           */
    def            onCyan = CYAN_B      + s + RESET
    
    /** Prints the background in ANSI WHITE.          */
    def           onWhite = WHITE_B     + s + RESET
    
    // Custom background colors.

    /** Prints the background in ANSI BRIGHT BLACK.   */
    def     onBrightBlack = B_BLACK_B   + s + RESET
    
    /** Prints the background in ANSI BRIGHT RED.     */
    def       onBrightRed = B_RED_B     + s + RESET
    
    /** Prints the background in ANSI BRIGHT GREEN.   */
    def     onBrightGreen = B_GREEN_B   + s + RESET
    
    /** Prints the background in ANSI BRIGHT YELLOW.  */
    def    onBrightYellow = B_YELLOW_B  + s + RESET
    
    /** Prints the background in ANSI BRIGHT BLUE.    */
    def      onBrightBlue = B_BLUE_B    + s + RESET
    
    /** Prints the background in ANSI BRIGHT MAGENTA. */
    def   onBrightMagenta = B_MAGENTA_B + s + RESET
    
    /** Prints the background in ANSI BRIGHT CYAN.    */
    def      onBrightCyan = B_CYAN_B    + s + RESET
    
    /** Prints the background in ANSI BRIGHT WHITE.   */
    def     onBrightWhite = B_WHITE_B   + s + RESET
  
    ///////////////////////////////////////////////////////////////////////////

    // Special text features.

    /** Prints the text in bold.                      */
    def              bold = BOLD        + s + RESET
    
    /** Prints the text underlined.                   */
    def        underlined = UNDERLINED  + s + RESET
    
    /** Prints the text blinking.                     */
    def             blink = BLINK       + s + RESET
  
    ///////////////////////////////////////////////////////////////////////////
   
    /** 
     *  Colorate numbers according to 2048 fashion.
     *  This method does not support non-digit input.
     *
     *  @throws TypeMismatchException 
     */
    def colorVal =
    {
      s.toInt match 
      {
        case    2 => s.white
        case    4 => s.cyan
        case    8 => s.green
        case   16 => s.blue
        case   32 => s.magenta
        case   64 => s.yellow
        case  128 => s.red
        case  256 => s.bold.brightCyan 
        case  512 => s.bold.brightBlue
        case 1024 => s.bold.brightYellow
        case 2048 => s.bold.brightRed.blink
        case    _ => s.white
      }
    }
  }
}

object Color extends Color
