### Practice 2. Ampliación de Programación Avanzada.

##### Pablo Acereda García and David Emanuel Craciunescu.

***

### Report in the assignment's proceedings

Just as the first assignment, the objective of the practice is to create the
well-known game **2048**. Although this time, the development is performed under
the Scala programming language. It has been pursued the knowledge of the
functinal programming.

### Assignment's insight

As for process carried out to develop de project, there have been followed the
following dogmas:

* Functional programming specified in the Scala documentation.
* The usage of inmutable fields and variables to avoid data corruption:
** Such us for example an inmutable list formed by inmutable values.
* An interface design utilizing console ASCII Art in order to get the user into
  a more inmersive visual experience.
* The utilization of a thorough planning, including details such as the progress
  obtained in the specific task, the developer of that task...
* A live system, to increase competitiveness of the users.

### Breakdown - Compulsory Assignment

The mandatory part of the assignment has been developed following the system
specifications:

* Game difficulty: Selected by the user at the beginning of the execution.
** From level 1 to level 4 (the lower the easier the game is).
** Each difficulty level has different board size.
* Board display: Also showing the different movements, lives available...
* Tiles merge: Each time two same-valued tiles collide one another, they are
  merged into one tile with a value equal to their addition.
** Tiles can only be merged once per movement.
* Merged-tiles count.
* Allowed movements: Must be checked whether the selected movement is valid.
* Game over: Just after the board is full, and no other movements can be
  performed.
* Points count: As it has been pointed before, each time two tiles merge, they
  also merge their values. That is going to be the method used to compute the
  points obtained.

### Breakdown - Optimized Assignmet

It has been a development decision to include the next perk to the system:

* As an improvement, it has been included a basic automated playing-mode system,
  in which the user does not need to intervene in order for the board to move.

