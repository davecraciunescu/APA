object aplicaLista
{
  def aplicaLista (lista: List[Int => Int], x: Int): Int =
  {
    if (lista.length == 1) lista.head(x);
    else lista.head(aplicaLista(lista.tail, x));
  }

  def max5 (x: Int): Int = x + 5;
  def por8 (x: Int): Int = x * 8;
  def suma3 (a: Int, b: Int, c: Int): Int = a + b + c;

  def main(args: Array[String]): Unit =
  {
    val l = List(max5 _, por8 _, suma3(1, _: Int, 10));
    println(aplicaLista(l, 10))
  }
}
