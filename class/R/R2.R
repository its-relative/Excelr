# Decision Making Statements in R
# If Statement
# Syntax: 
if (test_expression)
  {
  statement
  }

x <- -4
if(x > 0){
  print("Positive number")
  print(x)
}
# If Else Statement
# syntax: 
if (test_expression) {
  statement1
} else {
  statement2
  }

x <- 7
if(x > 0){
  print("Non-negative number")
} else {
  print("Negative number")
}

# Nested If Else Statement
if ( test_expression1) {
  statement1
} else if ( test_expression2) {
    statement2
  } else if ( test_expression3) {
  statement3
  } else  statement4

x <- -2
if (x == 0) 
  {
  print("Zero")
} else if (x > 0) 
  {
  print("Positive number")
} else
 print("Negative Number")

# There is an easier way to use if...else statement specifically for vectors in R programming
# Syntax: ifelse(test_expression,x,y)
a = c(5,7,2,9)

ifelse(a%%2==0 ,"even","odd")

a/2

a%%2
# for more than 2 conditions

client <- c("private", "public", "other",'private')

ifelse(client =='private', 1.12 , ifelse(client == 'other', 1.06, 1))

#ifelse(contition , true , ifelse(condition , 1 , ifelse(condition , 2 , 3)))
# Loops in R
# While loop
# Syntax
while (test_expression) {
  statement
}
#code
i <- 1
while (i<6) {
  print(i)
   i = i+1
}

  # Repeat loop
# Syntax 
repeat { 
  commands 
  if(condition) {
    break
  }
}
#code
v <- "Hello"
cnt <- 2

repeat {
  print(v)
  cnt <- cnt+1
  
  if(cnt > 5) {
    break
  }
}

# For Loop 
# Syntax
for (value in vector) {
  statements
}

v <- LETTERS[1:4]
v
for ( b in v) {
  print(b)
}

for (i in 1:length(v)){
  print(v[i])
}

# Loop Control Statements
# break Statement - terminates the loop statement and transfers execution to the statement 
# immediately following the loop
# next Statement - useful when we want to skip the current iteration of a loop without terminating it
v <- LETTERS[1:6]
for ( i in v) {
  
  if (i == "E") {
    next
  }
  print(i)
}

# Function 
function_name <- function(arg_1, arg_2, ...) {
  Function body 
}
# User-defined Function
new.function <- function(a) {
  for(i in 1:a) {
    b <- i^2
    print(b)
  }
}
# Calling a Function
new.function(10)

# Calling a Function without an Argument
hi <- function() {
  for(i in 1:5) {
    print(i^2)
  }
}	
hi()

# Calling a Function with Default Argument
# Create a function with arguments.
first <- function(a = 3, b = 6) {
  result <- a * b
  print(result)
}
# Call the function without giving any argument.
first()
# Call the function with giving new values of the argument.
first(5,7)

# Lazy Evaluation of Function
# Create a function with arguments.
new.function <- function(a, b) {
  print(a^2)
  print(a)
  print(b)
}
# Evaluate the function without supplying one of the arguments.
new.function(12,3)

# Mode in r

mySamples <- c(29, 4, 5, 7, 29, 19, 29, 13, 25, 19)
mean(mySamples)
median(mySamples)
mode(mySamples)
library(modeest)
mlv(mySamples, method = "mfv")
 

# Importing Cars dataset
data = read.csv(file.choose())
View(data)
summary(data)
data1 = read.csv(file.choose())

