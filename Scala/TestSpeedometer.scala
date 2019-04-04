package test.traits

object TestSpeedometer
{
  def main(args: Array[String]): Unit =
  {

    val speedometer1: Dashboard = new Dashboard(5)
    println("Initial Speed = " + speedometer1.showSpeed)
    speedometer1.accelerate(6)
    speedometer1.accelerate(9)

    val speedometer2: GenericDashboard[Double] = new GenericDashboard(25.1)
    println("Initial speed = " + speedometer2.showSpeed)
    speedometer2.accelerate(10.6)
    speedometer2.decelerate(5.9)
  }
}
