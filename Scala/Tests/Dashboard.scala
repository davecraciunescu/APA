package test.traits

class Dashboard (var speed: Float) extends Speedometer
{
  def showSpeed: Float = speed

  def accelerate (rate: Float) = println(f"Accelerating at $rate")

  def decelerate (rate: Float) = println(f"Decelerating at $rate")
}

class GenericDashboard[I] (var speed: I) extends GenericSpeedometer[I]
{
  def showSpeed: I = speed

  def accelerate (rate: I) = println(f"Accelerating at $rate")

  def decelerate (rate: I) = println(f"Decelerating at $rate")
}
