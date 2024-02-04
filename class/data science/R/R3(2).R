# Join (Merge) data frames (inner, outer, left, right) 
df1 = data.frame(CustomerId = c(1:6), Product = c(rep("Oven", 3), rep("Television", 3)))
df1
df2 = data.frame(CustomerId = c(2, 4, 6,7), State = c(rep("California", 3), rep("Texas", 1)))
df2

x=df1[2,]
x
rm(x)
x


# Merging happens based on the common column name in both the data sets
# Inner Join
df<-merge(df1,df2,by="CustomerId")
df
# Outer Join
df<-merge(x=df1,y=df2,by="CustomerId",all=TRUE)
df
x<-data.frame(CustomerId=c(9),Product=rep('mobile',1))

df3<-cbind(df,10:12) # it is used to combine rows in data frame
df3
# Left outer join
df<-merge(x=df1,y=df2,by="CustomerId",all.x=TRUE)
df
# Right outer join 
df<-merge(x=df1,y=df2,by="CustomerId",all.y=TRUE)
df
# Cross join
df<-merge(x = df1, y = df2, by = NULL)
df

df1 = data.frame(CustomerId = c(1,2,2), 
                 product=c('tv','AC','AC'))
df1
y=df1[-c(2),]
y
x=unique(df1)
x
x=duplicated(df1,all=TRUE)
x
uniq= df1[!duplicated(df1)]
uniq
data<-c(11,12,11)
duplicated(data)
data[duplicated(data)]
y=data[!duplicated(data)]
y
duplicated
data("iris")
View(iris)

# Apply functions 
# Returns a vector or array or list of values obtained by 
#applying a function to margins of an array or matrix or data frames 
#apply(df , 1 or 2 , function)

z = apply(iris[,1:4],2,mean)
z
# # lapply function takes list, vector or Data frame  as input and returns only list as output
#lapply(df , function)
x = lapply(iris[,1:4],mean)
x
# # sapply function takes list, vector or Data frame  as input. It is similar to lapply 
#function but returns only vector as output

y = sapply(iris[,1:4],mean)
y


data("iris")
iris$Sepal.Length

attach(iris)
Sepal.Length

# tapply
tapply(Sepal.Length, Species, mean) # mean of Sepal.Length for all 3 Species 

tapply(Sepal.Width, Species, median)
mean(iris$Sepal.Length)
data("mtcars")
View(mtcars)
x=mtcars$new.column=c(2,3,4)
x()
add_row()
nrow(mtcars) # no.of rows in mtcars
row.names(mtcars) # row names 
ncol(mtcars) # number of columns 
colnames(mtcars) # column names in mtcars 
names(mtcars)
dim(mtcars) # dimensions ( rows X columns )
dimnames(mtcars) # Dimension names ( row names and column names )

head(mtcars,10) 
tail(mtcars) 

data("mtcars")

min(mtcars$mpg) # na.rm --> Remove NA values 
max(mtcars$mpg, na.rm = TRUE)
range(mtcars$mpg,na.rm = T) # Return both min and max 

# mean, median
mean(mtcars$mpg, na.rm = T) # Average 
median(mtcars$mpg, na.rm = T) # middle most value in data after sorting in ascending or descending
mode(mtcars$hp)

library(modeest)
mlv(mtcars$hp, method = "mfv")
mtcars$cyl
x = factor(mtcars$cyl)
x
table(x)


##Handy dplyr Verb:
#Filter --> filter()
#Select --> select()
#Arrange --> arrange()

install.packages('dplyr')
library(dplyr)
#Structure:
# First Argument is a DataFrame
# Subsequent Argument say what to do with Data Frame
# Always return a Data Frame

#You can use "," or "&" to use and condition
#filter(df , condition )
mtcars[,c(2,10)]# extract entire rows and in columns 2 and 10 
filter(mtcars,cyl==8)

filter(mtcars,cyl==8,gear==5)


filter(mtcars,cyl==8 | gear == 5) # and gate (&) and OR gate(|)

#select method
 sel = select(mtcars,mpg,cyl,gear)
sel
filter(select(mtcars,mpg,cyl,gear),cyl == 4)
# Use ":" to select multiple contiguous columns, 
#and use "contains" to match columns by name

select(mtcars,carb,mpg:disp,gear)

#Syntax:
#arrange(dataframe,orderby)
arrange(select(mtcars,mpg,cyl),cyl)
arrange(mtcars,cyl)
arrange(select(mtcars,cyl,gear),cyl)
arrange(select(mtcars,cyl,gear),cyl,gear)
arrange(select(mtcars,cyl,gear),desc(cyl))
arrange(select(mtcars,cyl,gear),cyl,desc(gear))

# Visualizations
data("mtcars")
View(mtcars)
plot(mtcars$mpg,mtcars$disp) # Relationship between variables Scatterplot
plot(mtcars$disp,mtcars$hp)

# Histogram (Univariate)
hist(mtcars$mpg)

# Box Plot (To identify outliers)

boxplot(mtcars$hp)

install.packages("ggplot2")
library(ggplot2)
pairs(mtcars)
pairs(mtcars[,1:7])




