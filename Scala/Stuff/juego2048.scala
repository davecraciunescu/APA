
import scala.util.Random
import java.io.File;
import java.io.FileWriter;

object juego2048 {
  
  /*****TABLERO******/
  def main(args: Array[String]){
    println("------------ 1 6 3 8 4 ------------ ")
    println (" Este juego consiste en una versión del famoso 2048")
    println ("Para empezar a jugar, seleccione un nivel: ")
    println ("1.- FACIL \n2.- MEDIO \n3.- DIFICIL \n4.- EXPERTO")
    
    val dificultad = readInt()
    val tablero = tamTablero(dificultad)
    val filas = tablero._1
    val columnas = tablero._2
    val tableroVacio = List()
//(tablero:List[Int], filas:Int, columnas:Int, dificultad:Int)
    dificultad match{
     case 1=>jugar(crearTablero(filas*columnas),filas,columnas,dificultad)
     case 2=>jugar(crearTablero(filas*columnas),filas,columnas,dificultad)
     case 3=>jugar(crearTablero(filas*columnas),filas,columnas,dificultad)
    }
    
  }
  
  //función encargada de asingar unas dimensiones al tablero
  //en función de la dificultad elegida
  
  def tamTablero(dificultad:Int): (Int, Int) = dificultad match {
    case 1 => return (4,4)
    case 2 => return (9,9)
    case 3 => return (14,14)
    case 4 => return (17,17)
    //case default => println("La dificultad seleccionada no es válida")
  }
  //función ecaragada de crear un tablero vacío en función de la 
  //dificultad indicada
  def crearTablero(tam:Int):List[Int] = {
     if (tam == 0) return Nil
     else
       return 0::crearTablero(tam-1)
  }
  //Método que genera una semilla y la introduce en el tablero
  def generarSemillas(tablero:List[Int], dificultad:Int, posicion:Int):List[Int]= dificultad match{
    case 1 => {
      //val posicion = Random.nextInt(tam+1)
      if (tablero == Nil) return tablero
      else{
        if (posicion == 0) {
          return 2::tablero.tail
        }
        else{
          return tablero.head::generarSemillas(tablero, dificultad, posicion-1)
        }
      }
    }
  /*  case 2 => {
      val valores = 2* (Random.nextInt(2)+1) //valores 2,4
    }*/
  }
  
//  def generarSemillas(tablero:List[Int], tam:Int, dificultad:Int, semillas: Int):List[Int]= {
  //  val aux = random.nextInt(tam+1)
    //if (dificultad ==1){
      
    //}
  //}
    
  //Método para imprimir el número de las columnas 
  //con la finalidad de agilizar el testeo de datos
  def imprimir_cabeceras(dificultad:Int)= dificultad match{
    case 1 => {
      println("	1	2	3	4")
      println("	|	|	|	|")
    }
    case 2 =>{
      println("	1	2		3		4		5		6		7		8		9")
      println("	|	|		|		|		|		|		|		|		|")
    }
    case 3 =>{
      println("	1	2		3		4		5		6		7		8		9		10	11	12	13	14")
      println("	|	|		|		|		|		|		|		|		|		 |	 |	 |	 |	 |")
    }
    case 4 =>{
      println("	1	2		3		4		5		6		7		8		9		10	11	12	13	14	15	16	17")
      println("	|	|		|		|		|		|		|		|		|		 |	 |	 |	 |	 |	 |   |   |")
    }
  }
  
  def imprimir_tablero(filas:Int, fil:Int, columnas:Int, col:Int, dificultad:Int, tablero:List[Int]): Unit= {
    
    if(tablero == Nil) println("")
    else if(fil == 0){
      
      imprimir_cabeceras(dificultad)
      imprimir_tablero(filas, fil+1, columnas, col, dificultad, tablero)
      
    }
    else{
      
      if(col == 0){  //Caso primera columna
        
        if(fil <= filas && fil!= 0){
              printf((fil) + "---|   " + tablero.head + "   |")
              imprimir_tablero(filas, fil, columnas, col+1, dificultad, tablero.tail)
        }
        
      }
      else if(col == columnas-1){ //Caso última columna
        
        println("   " + tablero.head + "   |")
        imprimir_tablero(filas, fil+1, columnas, 0, dificultad, tablero.tail)
        
      }
      else{  //Resto de casos
        
        printf("   " + tablero.head + "   |")
        imprimir_tablero(filas, fil, columnas, col+1, dificultad, tablero.tail)
        
      }
      
    }
    
  }
  
 def dar_color(valores:Int):String ={
   if (valores == 2) return "A"
   else if (valores == 4) return "R"
   else if (valores == 8) return "N"
   else return "*"
 }
 
 def colorear(dificultad:Int):Int={
      if(dificultad == 1) return 4
      else if(dificultad == 2) return 5
      else  return 6
        
  }
 
/*def moverDer(tablero: List[Int], filas: Int, columnas: Int, posicion: Int):List[Int] ={
   
   if(tablero == Nil) return Nil
   else if((posicion+1) % columnas == 0){
     return tablero.head :: moverDer(tablero.tail, filas, columnas, posicion+1)
   }
   else if(tablero.head != 0 && tablero.tail.head == 0){
     
     val aux = (tablero.head::tablero.tail.tail)
     val tab2 = 0 :: moverDer(aux, filas, columnas, posicion+1)
     return moverDer(tab2, filas, columnas, posicion)
     
   }
   else if(tablero.head != 0 && tablero.tail.head == tablero.head){
     
     val num = tablero.head * 2
     val tab2 =  0 :: moverDer(num::tablero.tail.tail, filas, columnas, posicion+1)
     return moverDer(tab2, filas, columnas, posicion)
   
   }
   else return tablero.head :: moverDer(tablero.tail, filas, columnas, posicion+1)
   
 }*/
 
 def moverDer(tablero: List[Int], filas:Int,columnas:Int, posicion:Int):List[Int] = {
   if (tablero==Nil) return Nil
   else if((posicion+1) % columnas == 0){
     return tablero.head :: moverDer(tablero.tail, filas, columnas, posicion+1)
   }
   else if( coger(posicion,tablero)!=0 && coger(posicion+1, tablero)== 0){
      val aux = (tablero.head::tablero.tail.tail)
      val tab2 = 0 :: moverDer(aux, filas, columnas, posicion+1)
      return moverDer(tab2, filas, columnas, posicion)
   }
   else if(coger(posicion,tablero) != 0 && (coger(posicion+1,tablero) == coger(posicion,tablero))){
     
     val num = tablero.head * 2
     val tab2 =  0 :: moverDer(num::tablero.tail.tail, filas, columnas, posicion+1)
     return moverDer(tab2, filas, columnas, posicion)
   
   }
   else if ( coger(posicion,tablero)!=0 && coger(posicion+1, tablero)!= 0){
     return 0::moverDer(tablero, filas, columnas, posicion+2)
   }
   else if (coger (posicion, tablero)==0) return moverDer(tablero.tail, filas, columnas, posicion+1)
    else return tablero.head :: moverDer(tablero.tail, filas, columnas, posicion+1)
 }
def moverIzq(tablero: List[Int], filas: Int, columnas: Int, posicion: Int):List[Int] ={
   
    return moverDer(tablero.reverse, filas, columnas,0).reverse
}
  
 def traspuesta (tablero:List[Int], columnas: Int, tam:Int, pos:Int):List[Int]={
   if (tablero == Nil)
     return tablero
   else{
     if (pos>=tam) return traspuesta(tablero.tail, columnas, tam-1, 0)
     else if (tam == columnas) return Nil
     else coger(pos, tablero)::traspuesta(tablero, columnas, tam, pos+columnas)
   }
 }
 
 def coger(n:Int, tablero:List[Int]):Int={
   if (n==0) return tablero.head
   else coger(n-1, tablero.tail)
 }
 
 def mover (movimiento:Int, tablero:List[Int], filas:Int, columnas:Int) = movimiento match {
   case 2 =>
   case 4 => 000000..0.0.00000
   case 6 => moverDer(tablero, filas, columnas,0)
   case 8 =>
   case _ =>
 }
  
 //método principal del juego
  def jugar(tablero:List[Int], filas:Int, columnas:Int, dificultad:Int) = dificultad match{
    case 1 =>{
      val tablerogen = List(2,4,8,0,2,4,8,0,2,4,8,0,2,4,8,0)
      
      //val tablerogen = generarSemillas(tablero, dificultad, Random.nextInt(filas*columnas))
      imprimir_tablero(filas,0,columnas,0, dificultad, tablerogen)
      //println("___________")
      //println(generarSemillas(tablero, dificultad,  3))
      if (tablero != Nil){
        //Realizar movimiento + llamada recursiva a jugar
        //val movimiento = readInt()
        //mover(movimiento, tablero, filas, columnas)
        println("Mover derecha")
        imprimir_tablero(filas, 0, columnas, 0, dificultad, moverDer(tablerogen, filas, columnas, 0))
        //mover(movimiento, tablero, filas, columnas)
        println ("Matriz izquierda")
        val prueba= traspuesta(tablerogen, columnas, columnas*filas, 0)
        imprimir_tablero(filas, 0, columnas, 0, dificultad, traspuesta(prueba, columnas, columnas*filas, 0))
        //val prueba = List(0,2,4,2)
        //println(prueba.tail.tail.head)
        //println(coger(4, tablero))
        
        //println(traspuesta(prueba,columnas,columnas*filas, 0))
        //traspuesta(prueba,columnas,columnas*filas, 0))
        //imprimir_tablero(filas, 0, columnas, 0, dificultad, moverDer(traspuesta(prueba,columnas,columnas*filas, 0), filas, columnas, 0))
      }
    
    }
  }
}