object movimientos
{
  /**
   * As all the movements are based on the right movement, the structure of a
   * movement is generalized so there is only needed one method.
   *
   * @param board List which contains the tiles.
   * @param col   Number of columns.
   *
   * @return Board having moved. 
   *
   */
  private def moverGen(tablero: List[Int], columnas: Int, posicion: Int):List[Int] =
  {
    val m1 = moverDer(tablero, columnas, 0)
    val sum = sumar(m1, columnas, 0)
    val m2 = moverDer(sum, columnas, 0)
    m2
    
  }
 
  /**
   * Moves a matrices tiles to the right. 
   * Tiles with a 0 count as empty.
   *
   * @param board The List which holds the tiles.
   * @param col   Number of columns. 
   * @param pos   Position to iterate whithin the board.
   *
   * @return Move the board to the right.
   *
   */
 private def moverDer (tablero: List[Int], columnas:Int, posicion:Int):List[Int] = 
 {
    if (tablero != Nil) moverDerAux(cogerN(columnas,tablero), columnas, posicion):::moverDer(quitar(columnas, tablero), columnas, posicion);
    else tablero
 }
  
  /**
   * Recursive method to iterate thorough me matrix and move the tiles to the
   * right.
   *
   * @param board The List which holds the tiles.
   * @param col   Number of columns. 
   * @param pos   Position to iterate whithin the board.
   *
   * @return Element moved to the right.
   *
   */
private def moverDerAux(tablero: List[Int], columnas: Int, posicion: Int):List[Int] =
{
   if(tablero.tail == Nil) tablero
   else if((posicion + 1 ) % columnas == 0)
   {
     tablero.head :: moverDer(tablero.tail, columnas, posicion+1)
   }
   else if(tablero.head != 0 )
   {
     if (tablero.tail.head == 0)
     {
       0 :: moverDerAux(tablero.head :: tablero.tail.tail, columnas, posicion + 1)
     }
     else
     {
       if (tablero.reverse.head == 0) 
         0 :: moverDerAux((tablero.reverse.tail).reverse, 
                          columnas, posicion + 1)
       else tablero.head :: moverDerAux(tablero.tail, columnas, posicion + 1)
     } 
   }
   else
   {
     tablero.head :: moverDerAux(tablero.tail, columnas, posicion + 1)
   }
 }

  /**
   * Merges contiguos tiles if the have the same value. 
   * The resulting tile has the addition of the values of its parents tiles.
   * The tiles merged cannot be merged more than once in the same movement.
   * 
   * @param board The List which holds the tiles.
   * @param col   The number of columns. 
   * @param pos   Position to iterate whithin the board.
   *
   * @return Tiles merged.
   *
   */
def sumar(tablero: List[Int], columnas: Int, posicion: Int): List[Int] =
{
  if(tablero.tail == Nil) tablero
  else if((posicion + 1) % columnas == 0)
  {
     tablero.head :: sumar(tablero.tail, columnas, posicion+1)
  }
  else
  {
    if(tablero.head != 0 && tablero.tail.head == tablero.head){
      val sum = tablero.head * 2
      val tab = sum :: tablero.tail.tail
      if(tab.tail == Nil) 0 :: tab
      else
      {
        if(tab.head == tab.tail.head) 0 :: tab.head :: sumar(tab.tail, columnas, posicion + 2)
        else 0 :: sumar(tab, columnas, posicion+1)
      }
    }
    else tablero.head :: sumar(tablero.tail, columnas, posicion + 1)
  }
} 

  /**
   * Moves the tiles of a matrix to the left.
   *
   * @param board The List which holds the tiles.
   * @param col   Number of columns.
   *
   * @return Board moved to the left.
   *
   */
  def moverIzq(tablero: List[Int], columnas: Int, posicion: Int): List[Int] =
  {  
    // The method is actually the movement to the right of the reversed
    // matrix. 
    // The matrix again reversed to obtain the original matrix.
    moverGen(tablero.reverse, columnas, 0).reverse
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
   *
   */
  def moverAbajo(tablero: List[Int], columnas: Int, tam: Int, posicion: Int): List[Int] = 
  {
    // It uses a transposed matrix to fill move the tiles.
    // After the final matrix is obtained, it is then again transposed to obtain
    // the original matrix.
    val tras = traspuesta(tablero, columnas, tam, posicion)
    val mov  = moverDer(tras, columnas, posicion)

    traspuesta(moverGen(traspuesta(tablero, columnas, tam, posicion), columnas, posicion), columnas, tam, posicion)
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
   *
   */
  def moverArriba(tablero: List[Int], columnas: Int, tam: Int, posicion: Int): List[Int] =
  {
    // It generates the transposed matrix and sends it to the left movement.
    // The transposed of the list obtained by that method is the original
    // matrix.
    val tras = traspuesta(tablero, columnas, tam, posicion)
    val mov = moverIzq(tras, columnas, posicion)
   
    traspuesta(moverIzq(traspuesta(tablero, columnas, tam, posicion), columnas, posicion), columnas, tam, posicion)
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
   *
   */
  def traspuesta (tablero: List[Int], columnas: Int, tam: Int, pos: Int): List[Int] = 
  {
    if (tablero == Nil) tablero
    else
    {
      if (tam == columnas) Nil
      else if (pos >= tam) traspuesta(tablero.tail, columnas, tam - 1, 0)
      else coger(pos, tablero) :: traspuesta(tablero, columnas, tam, pos + columnas)
    }
  }
 
  /**
   * Retrieves the first n elements from a list.
   *
   * @param n Position.
   * @param l List which contains the tiles. 
   *
   * @return First n elements.
   *
   */
  def cogerN(n: Int, l: List[Int]): List[Int] =
  {
   if (n == 0) Nil
   else        l.head :: cogerN(n - 1, l.tail)
 }
 
  /**
   * Removes the first n elements from a list.
   *
   * @param n Position.
   * @param l List which contains the tiles.
   *
   * @return List without first n elements.
   *
   */
 def quitar(n: Int, l: List[Int]): List[Int] =
 {
   if (l == Nil) Nil
   else if (n == 0) l
   else quitar(n - 1, l.tail)
 }
 
  /**
   * Retrieves the specified element.
   *
   * @param n     Position.
   * @param board List which contains the tiles.
   *
   * @return Element n.
   *
   */
 def coger(n: Int, tablero: List[Int]): Int=
 {
   if (n == 0) tablero.head
   else coger(n - 1, tablero.tail)
 }
 
  def mover (movimiento: String, tablero: List[Int], columnas: Int): List[Int] =
    movimiento match
  {
    case ("s"|"S") => moverAbajo(tablero, columnas, columnas * columnas, 0)
    case ("a"|"A") => moverIzq(tablero, columnas, 0)
    case ("d"|"D") => moverGen(tablero, columnas, 0)
    case ("w"|"W") => moverArriba(tablero, columnas, columnas * columnas, 0)
  }
}
