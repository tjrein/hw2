==== CS613 ====

  Author: Tom Rein
  Email: tr557@drexel.edu

==== Dependencies ====
  * python3
  * pip3
  * numpy
  * matplolib.pyplot

  In the event that dependencies are not installed, I have provided a "requirements.txt" file.
  To install dependencies, type "pip3 install -r requirements.txt"


==== Files Included ====
  * data_operations.py
  * lr_global.py
  * lr_local.py
  * s_folds.py
  * gradient_descent.py
  * x06Simple.csv

  NOTE: x065Simple.csv needs to be in the present working directory.
        I have included the file in the .zip, so this should work out of the box


==== data_operations.py ====
  Contains shared functions used for manipulating the x06Simple data set.
  These functions are used by the other scripts.


==== lr_global.py ====

  This script performs global/closed-form linear regression on the x06Simple data set.
  It will output the theta values used for the final model as well as the RMSE to the console.

  To execute, type "python3 lr_global.py"


==== lr_local.py ====

  This script performs local liner regression one the x06Simple data set.
  It will output the RMSE to the console

  To execute, type "python3 lr_local.py"


==== s_folds.py ====

  This script performs S-Folds cross validation on the x06Simple data set.
  The script can be initialized with an optional integer parameter to specify the number of folds.

  To execute, type "python3 s_folds.py {s}" where s is an integer.
  If no argument is passed, the script will default to using 3 folds.

  The script will output the final RMSE and standard deviation of performing S-Folds 20 different times.


==== gradient_descent.py ===

  This script performs gradient descent on the x06Simple data set.
  It will output the final theta values and the final RMSE after reaching the termination condition.
  The script will also display a plot of the RMSE of the training and testing sets with respect to the number of iterations.
  Finally, the script will save an image of the plot as "gradient_descent.png"

  To execute, type "python3 gradient_descent.py"
