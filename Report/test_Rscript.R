weight<-read.csv("WeightClassification.csv")
micro<-read.csv("gastro_microscope.csv")

x<-table(weight$BMI)
barplot(x,las=2)
