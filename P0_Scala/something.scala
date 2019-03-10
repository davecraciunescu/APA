object prueba3 {
  println("Welcome to Scala worksheet")
  1+3

  def max(x: Int, y: Int): Int = {
        if(x > y) x
        else      y
  }

  max(7,9)

  val a = List()
  val b = List()
  val c = List()
    
  val d = b:::c
        
  b::c    

  c.head
  c.tail
  
  def numElems(lista:List[Int]):Int =
    if(lista.isEmpty)
      0
    else
      1 + numElems(lista.tail)
  def numElems(lista:List[Int]):Int =
        if(lista.isEmpty)
          0
        else
          lista.head + sumElems(lista.tail)
 
  var x = numElems(d)

  val y = 5
  
  def factorial (n:Int):Int = n match {
    case 0 
    case _
  }
}
