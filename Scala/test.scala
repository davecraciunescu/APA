object test extends Application
{
  def max(x: Int, y:Int): Int =
  {
    if (x > y) x
    else y
  }

  def numElems(list:List[Int]):Int =
    if (list.isEmpty)
      0
    else
      1 + numElems(list.tail)

  def sumElems(list:List[Int]):Int =
    if (list.isEmpty)
      0
    else
      list.head + sumElems(list.tail)

  def sumaZeroCien(n:Int):Int = n 
    match
    {
      case 1 => 1
      case n => n + sumaZeroCien(n-1)
    }

  max(7,9)
}



