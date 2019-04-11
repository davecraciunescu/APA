object movimientos
{
  def moverGen(tablero: List[Int], columnas: Int, posicion: Int):List[Int] =
  {
    val m1 = moverDer(tablero, columnas, 0)
    val sum = sumar(m1, columnas, 0)
    val m2 = moverDer(sum, columnas, 0)
    m2
    
  }
 
 def moverDer (tablero: List[Int], columnas:Int, posicion:Int):List[Int] = 
 {
    if (tablero != Nil) moverDerAux(cogerN(columnas,tablero), columnas, posicion):::moverDer(quitar(columnas, tablero), columnas, posicion);
    else tablero
 }
  
def moverDerAux(tablero: List[Int], columnas: Int, posicion: Int):List[Int] =
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
  def moverIzq(tablero: List[Int], columnas: Int, posicion: Int): List[Int] =
  {  
    moverGen(tablero.reverse, columnas, 0).reverse
  }

  def moverAbajo(tablero: List[Int], columnas: Int, tam: Int, posicion: Int): List[Int] = 
  {
    val tras = traspuesta(tablero, columnas, tam, posicion)
    val mov  = moverDer(tras, columnas, posicion)

    traspuesta(moverGen(traspuesta(tablero, columnas, tam, posicion), columnas, posicion), columnas, tam, posicion)
  }

  def moverArriba(tablero: List[Int], columnas: Int, tam: Int, posicion: Int): List[Int] =
  {
    val tras = traspuesta(tablero, columnas, tam, posicion)
    val mov = moverIzq(tras, columnas, posicion)
   
    traspuesta(moverIzq(traspuesta(tablero, columnas, tam, posicion), columnas, posicion), columnas, tam, posicion)
  }

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
 
  def cogerN(n: Int, l: List[Int]): List[Int] =
  {
   if (n == 0) Nil
   else        l.head :: cogerN(n - 1, l.tail)
 }
 
 def quitar(n: Int, l: List[Int]): List[Int] =
 {
   if (l == Nil) Nil
   else if (n == 0) l
   else quitar(n - 1, l.tail)
 }
 
 def coger(n: Int, tablero: List[Int]): Int=
 {
   if (n == 0) tablero.head
   else coger(n - 1, tablero.tail)
 }
 
 def mover (movimiento: Int, tablero: List[Int], columnas: Int, dificultad: Int) = 
   movimiento match 
   {
     case 2 => moverAbajo(tablero, columnas, columnas * columnas, 0)
     case 4 => moverIzq(tablero, columnas, 0)
     case 6 => moverGen(tablero, columnas, 0)
     case 8 => moverArriba(tablero, columnas, columnas * columnas, 0)
     case _ => println ("Movimiento no válido")
   }
 }
