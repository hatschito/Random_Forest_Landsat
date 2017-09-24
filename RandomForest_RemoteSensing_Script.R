#########################################################
#Script: Harald Schernthanner: RandomForest classifier /#  
#Author: Schernthanner, Cardenas                        #
#  Ver.: 0.3                                            #
#########################################################

##########################################
#Content                                 #
#I: Perparing the R working enviroment   #
#II: Preprocess the imagery              #
#III: RandomForest Model                 #
#IV:Accuracy assessment                  #
#V: Credits                              #
##########################################

#Complementary code (e.g. preprocessing steps) preprocessing is commented out 

#########################################################
#I: Perparing the R working enviroment                  #
# Install packagea, load necessary libraries            #
#########################################################

#Clear Workspace
rm(list = ls())

#System path to the data
#setwd("G:/DOKUMENTE/Fuer Auras Dissertation/Aura_Daten/")
#Set working dirctory on Mac
setwd("/Volumes/Harald_1/DOKUMENTE/Fuer Auras Dissertation/")

#List all Data in Workspace
ls()
#Get path of the working directory
getwd() 

#Save and load entire workspace
save.image("Aura_Analysis.RData")
load("Aura_Analysis.RData")

#Reporting the maximum memory size of the OS
#memory.size(max = TRUE)

#Install and load necessary libraries/packages#

#rgdal - import, export raster
install.packages("rgdal")
library(rgdal)
#check gdal version number
#getGDALVersionInfo()

#RandomForest 
#install.packages("randomForest")
library(randomForest)

#Load R raster package
#install.packages("raster")
library(raster)

#RStoolbox
install.packages("RStoolbox")
library(RStoolbox)

#SP: spatial data handling
#Load library sp
library(sp)

#R sampling 
install.packages("sampling")
library(sampling)

#Package for validation / confusion matrix
#install.packages("asbio")
library(asbio)

#########################################################
#II Preprocess the imagery                              #   
#Most pre-processing (e.g. subsetting) was done in QGIS-#
#(Semi automatic classification plugin)                 #
##########################################################

#Import the Landsat satellite imagery 
Landsat_8_UTM <- brick('Rasterdaten/Matiguas_Landsat2015.tif')

#Landsat DOS2 atmospheric correction with RStoolbox in case the correction wasn´t done in QGPS(SLCP plugin)
#Path to Landsat Metadata
#xml_meta <- readMeta("Rasterdaten/Matiguas_Landsat2015.tif.aux.xml)
#haze <- estimateHaze(img, darkProp =0.01)
#img_sdos <- radCor(img, hazeValues = haze, metaData = xml_meta, method = "sdos")


#Reprojecting the dataset to LatLong 
#Landsat8_lat_long <- projectRaster(Landsat_8, crs='+proj=longlat')
#Rescaling is just necessary in case the data hasn´t absolute reflectence values 
Landsat_Matiguas_rescale <- calc(Landsat_8_UTM, fun=function(x) x / 10000)

#Load Training polygons and save them as .rda files 
trainingPoly_UTM <- readOGR('Vektordaten/Trainingsgebiete/Trainings areas 25_5_2016.shp', layer = 'Trainings areas 25_5_2016')


#Get the reference system
#trainingPoly <- trainingPoly_UTM
#Reproject to lat long
#crs(trainingPoly) <- '+proj=longlat'
#Check projection again 
#crs(trainingPoly)

#Plot imagery in true/false color 
#hist(Landsat_8_UTM)
plotRGB(Landsat_8_UTM, 3,2,1)

#Plot training areas
plot(trainingPoly_UTM, add = TRUE)

#Check the training data
trainingPoly_UTM@data
plot(trainingPoly_UTM)
str(trainingPoly_UTM)


#Rasterize the trainig data
classes <- rasterize(trainingPoly_UTM, Landsat_8_UTM, field='MC_ID')

#Histogramm and plot of training data
hist(classes)
cols <- c("dark green", "yellow", "orange", "blue", "white", "black", "red")
plot(classes, col=cols, legend=FALSE)


#Masking the satellite imagery by the training data 
trainmask <- mask(Landsat_8_UTM, classes)

#plot(trainmask)
names(classes) <- "class"
train_samples <- addLayer(trainmask, classes)
#plot(trainingbrick)
#Getting the values

#Resulting matrix is very large ~ 3-4 GB. Workaround, export the resulting dataset and remove NAs with Unix/grep 
#Reason for the large file size, the every cell without spectral reflactance value that doesnt correspond with the
#spectral samples, is filled by NAs(No data values)

valuetable <- getValues(train_samples)

write.csv(valuetable, "valuetable.csv")
#Grep in Unix: grep -Ev NA valuetable.csv  > valuetable_withoutNA.csv
#valuetable <- read.csv("valuetable_withoutNA.csv")

valuetable <- na.omit(valuetable)
valuetable <- as.data.frame(valuetable)
head(valuetable, n = 100)
#tail(valuetable, n = 100)
str(valuetable)


##########################################################
#III RandomForest Model                                  #   
##########################################################

############################
####Preamble to the model###
###########################

#RandomForestModel - Model training:  with n trees / x corresponds to the number of spectral bands, y to the thematic classes
#RandomForest packages makes use of fortran and only accepts intege numbers, so long vectors may be an issue and requiere a workaround
#number of observations has to be lower than than 2^ 31- (4.294.967.296 values) - our observation exceed the values so 
#Issue is documentad on:  https://stackoverflow.com/questions/22454868/r-how-to-use-long-vectors-with-randomforest

#Setting the maximum possible dimensions of the training data
#maxDim <- 2^31 - 1;
#train_samples_maxDim <- train_samples[1:maxDim, ]
#another approach: random sampling the dataframe holding the trainingdata
#Random sampling, keeping 25% of the exisiting raws
valuetable_sampled <- valuetable[sample(nrow(valuetable),replace=FALSE,size=0.25*nrow(valuetable)),]

#Tune the model with the integrated tuneRF function

#Independend variable 
x <- valuetable_sampled[ ,c(1:6)]

#Dependend variable to be classified
y <- valuetable_sampled$layer

tuneRF(x, y,
       mtryStart = 3, 
       ntreeTry=1, 
       stepFactor = 0.1, 
       improve = 0.0001, 
       trace=TRUE, 
       plot = TRUE,
       doBest = TRUE,
       nodesize = 30, 
       importance=TRUE
       )


#mtry = Number of variables available for splitting at each tree node /very sensible model parameter
#Trainining stage using tuned model parametres 
modelRF <- randomForest(x, y, importance = TRUE, mtry = 3, ntree = 500,  do.trace = 100)


####Parameters used to tune our RF model and to adapt it to to our data: 
#do.trace, traces the out of the bag error for different tree numbers 
#cutoff, cuts the tree of, if e.g. 700 from 1000 trees make the same class decision, e.g. cutoff = c(0.7, 0.3
# ntrees: Numbrt of RandomFores trees 


modelRF
class(modelRF)
str(modelRF)
names(modelRF)

#plot confusion matrix
modelRF$confusion

colnames(modelRF$confusion) <- c("Forest", "Pasture", "Degraded Pasture", "Water", "Cloud", "Shadow", "Urban","class.error")
rownames(modelRF$confusion) <- c("Forest", "Pasture", "Degraded Pasture", "Water", "Cloud", "Shadow", "Urban")

modelRF$confusion
varImpPlot(modelRF)


#Classification based on the trained model parametes
classification <- predict(Landsat_8_UTM, model=modelRF, na.rm=TRUE)
writeRaster(classification, filename = "Random_Forest_20_9.tif",overwrite = TRUE, datatype = 'INT2S')

#########################################################
#IV Accuracy Assessment                                 #   
##########################################################

sample_reference <- read.csv2("Sample_reference.csv", sep = ",")
sample <- as.data.frame(sample_reference)

landsat<-sample$X1
reference<-sample$X1.1

#Transpone data
landsat.t <- t(landsat)
reference.t<-t(reference)

#calculate the kappa 
asbio::Kappa(landsat,reference) 





##############################################################################################################################
#CV credits:                                                                                                                 #
#   R Raster package: Hijmans et al: https://cran.r-project.org/web/packages/raster/index.html                               #
#   RandomForest: A. Liaw and M. Wiener (2002). Classification and Regression by randomForest. R News 2(3), 18--22.          #
#   Model tuning: https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/                                  #   
#   Data frame manipulation: https://www.rdocumentation.org/packages/fifer/versions/1.0                                      #
#   RStoolbox: Benjamin Leuner and Net Horning: https://cran.r-project.org/web/packages/RStoolbox/index.html                 #
#   R sp package: Pepebsma and Hijmans:https://cran.r-project.org/web/packages/sp/index.html                                 #
#   asbio: A Collection of Statistical Tools for Biologists, Aho, https://cran.r-project.org/web/packages/asbio/index.html   #
##############################################################################################################################



####Here comes the external accuracy assessment