  package test.traits

trait Speedometer
{
  protected var speed: Float
  def       showSpeed: Float
  def accelerate(rate: Float)
  def decelerate(rate: Float)
}

trait GenericSpeedometer[I]
{
  protected var speed: I
  def showSpeed: I
  def accelerate(rate: I)
  def decelerate(rate: I)
}
