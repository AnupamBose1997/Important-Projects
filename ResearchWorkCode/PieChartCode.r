library("xlsx") 
library(dplyr)
library(cowplot)
library(ggpubr)
library(tidyverse)
library(RColorBrewer)

####################################################
##Family Level Pie Chart - Group X
##For Day-1,Day-2,Day-5 

p=read.xlsx2("OTU_Drug_Data_Bookchap.xlsx",sheetIndex = 1)


p%>%
  select(-c(1)) %>%
  select(-c(3)) %>%
  filter(p$Group == 'X' & p$Timepoint == 'D1') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp1
names(temp1)[2]<-paste("D1")
temp1 %>%
  mutate(D1=round(D1,digits = 2)) -> temp1

p%>%
  select(-c(1))%>%
  select(-c(3))%>%
  filter(p$Group == 'X' & p$Timepoint == 'D2') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp2
names(temp2)[2]<-paste("D2")
temp2 %>%
  mutate(D2=round(D2,digits = 2)) -> temp2


p%>%
  select(-c(1))%>%
  select(-c(3))%>%
  filter(p$Group == 'X' & p$Timepoint == 'D5') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp5
names(temp5)[2]<-paste("D5")
temp5 %>%
  mutate(D5=round(D5,digits = 2)) -> temp5

inner_join(temp1,temp2,by=c('Dataset')) %>%
  inner_join(temp5,by=c('Dataset')) -> InitialDays

InitialDays[is.na(InitialDays)] <- 0  

#average
InitialDays %>%
  mutate(Values = round((D1+D2+D5)/3,digits = 2)) -> xInitialDays

#top 10 and 'others': percentage
xInitialDays %>%
  select(Dataset,Values) %>%
  filter(Values!=0) %>%
  mutate(Values=round(as.numeric(Values),digits=2)) %>%
  mutate(per=round(Values/sum(Values)*100,digits = 2)) %>%
  arrange(desc(per)) ->xInitialDays

xInitialDays <- rbind(xInitialDays,c(NA,colSums(xInitialDays[c(11:64),sapply(xInitialDays,is.numeric)])))
xInitialDays$Dataset[65] <- "Others"
xInitialDays <- xInitialDays[c(1:10,65),]

#version 1 - choosing colors
#allcols=kelly(n=20)
#allcols<-allcols[c(2:20)]

#version 3 - choosing colors
allcols= 20
getPalette = colorRampPalette(brewer.pal(12, "Set3"))
fill=getPalette(allcols)
allcols<-fill

#version 2 - choosing colors
#allcols <- c('#2166ac','#5e4fa2','#878787','#abd9e9','#3288bd','#ffed6f','#dfc27d','#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5')

xInitialDays %>%
  mutate(taxa = xInitialDays$Dataset) %>%
  mutate(cols = allcols[c(1:11)]) %>%
  select(taxa,cols) -> ColAssgn
len = 11

xinidays <- function(){
xInitialDays%>%
  ggplot(aes(x='',y=reorder(Values,Values),fill=Dataset))+
  geom_bar(width=1,stat="identity")+
  scale_fill_manual(breaks = ColAssgn$taxa,
                      values = ColAssgn$cols)+
  theme_void()+
  theme_classic() +
  theme(legend.position = "top") +
  coord_polar("y",start=0) +
  theme(axis.line = element_blank())+
  theme(axis.text = element_blank()) +
  theme(axis.ticks = element_blank())+
  labs(x = NULL, y = NULL, fill = NULL)+
  ylab("Baseline")
}



#############################################################
#For Day-7,Day-10,Day-15


p%>%
  select(-c(1)) %>%
  select(-c(3)) %>%
  filter(p$Group == 'X' & p$Timepoint == 'D7') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp7
names(temp7)[2]<-paste("D7")
temp7 %>%
  mutate(D7=round(D7,digits = 2)) -> temp7

p%>%
  select(-c(1))%>%
  select(-c(3))%>%
  filter(p$Group == 'X' & p$Timepoint == 'D10') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp10
names(temp10)[2]<-paste("D10")
temp10 %>%
  mutate(D10=round(D10,digits = 2)) -> temp10


p%>%
  select(-c(1))%>%
  select(-c(3))%>%
  filter(p$Group == 'X' & p$Timepoint == 'D15') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp15
names(temp15)[2]<-paste("D15")
temp15 %>%
  mutate(D15=round(D15,digits = 2)) -> temp15

inner_join(temp7,temp10,by=c('Dataset')) %>%
  inner_join(temp15,by=c('Dataset')) -> MidDays

MidDays[is.na(MidDays)] <- 0  

MidDays %>%
  mutate(Values = round((D7+D10+D15)/3,digits = 2)) -> xMidDays


xMidDays %>%
  select(Dataset,Values) %>%
  filter(Values!=0) %>%
  mutate(Values=round(as.numeric(Values),digits=2)) %>%
  mutate(per=round(Values/sum(Values)*100,digits = 2)) %>%
  arrange(desc(per)) -> xMidDays

xMidDays <- rbind(xMidDays,c(NA,colSums(xMidDays[c(11:64),sapply(xMidDays,is.numeric)])))
xMidDays$Dataset[65] <- "Others"        
xMidDays <- xMidDays[c(1:10,65),]


taxalen <- xMidDays$Dataset[c(which(!(xMidDays$Dataset %in% ColAssgn$taxa)))]

for (count in 1:length(taxalen)){
  if(length(taxalen) > 0){
    len = len +1
    ColAssgn %>%
      rbind(c(taxalen[count],allcols[len])) -> ColAssgn
    
  }
}



xmiddays <- function(){
xMidDays %>%
  ggplot(aes(x='',y=reorder(Values,Values),fill = Dataset))+
  geom_bar(width=1,stat="identity")+
  scale_fill_manual(breaks = ColAssgn$taxa,
                      values = ColAssgn$cols)+
  theme_void()+
  theme_classic() +
  theme(legend.position = "top") +
  coord_polar("y",start=0) +
  theme(axis.line = element_blank())+
  theme(axis.text = element_blank()) +
  theme(axis.ticks = element_blank())+
  labs(x = NULL, y = NULL, fill = NULL)+
  ylab("Mid Treatment")

}

#######################################################################
##For Day-20,Day-25,Day-30


p%>%
  select(-c(1)) %>%
  select(-c(3)) %>%
  filter(p$Group == 'X' & p$Timepoint == 'D20') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp20
names(temp20)[2]<-paste("D20")
temp20 %>%
  mutate(D20=round(D20,digits = 2)) -> temp20

p%>%
  select(-c(1))%>%
  select(-c(3))%>%
  filter(p$Group == 'X' & p$Timepoint == 'D25') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp25
names(temp25)[2]<-paste("D25")
temp25 %>%
  mutate(D25=round(D25,digits = 2)) -> temp25


p%>%
  select(-c(1))%>%
  select(-c(3))%>%
  filter(p$Group == 'X' & p$Timepoint == 'D30') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp30
names(temp30)[2]<-paste("D30")
temp30 %>%
  mutate(D30=round(D30,digits = 2)) -> temp30

inner_join(temp20,temp25,by=c('Dataset')) %>%
  inner_join(temp30,by=c('Dataset')) -> Finaldays

Finaldays[is.na(Finaldays)] <- 0  

Finaldays %>%
  mutate(Values = round((D20+D25+D30)/3,digits = 2)) -> xFinaldays


xFinaldays %>%
  select(Dataset,Values) %>%
  filter(Values!=0) %>%
  mutate(Values=round(as.numeric(Values),digits=2)) %>%
  mutate(per=round(Values/sum(Values)*100,digits = 2)) %>%
  arrange(desc(per)) -> xFinaldays
    
xFinaldays <- rbind(xFinaldays,c(NA,colSums(xFinaldays[c(11:51),sapply(xFinaldays,is.numeric)])))
xFinaldays$Dataset[52] <- "Others"        
xFinaldays <- xFinaldays[c(1:10,52),] 



taxalen <- xFinaldays$Dataset[c(which(!(xFinaldays$Dataset %in% ColAssgn$taxa)))]

for (count in 1:length(taxalen)){
  if(length(taxalen) > 0){
    len = len +1
    ColAssgn %>%
      rbind(c(taxalen[count],allcols[len])) -> ColAssgn
    
  }
}

xfindays <- function(){
xFinaldays %>%
  ggplot(aes(x='',y=reorder(Values,Values),fill = Dataset))+
  geom_bar(width=1,stat="identity")+
  scale_fill_manual(breaks = ColAssgn$taxa,
                      values = ColAssgn$cols)+
  theme_void()+
  theme_classic() +
  theme(legend.position = "top") +
  coord_polar("y",start=0) +
  theme(axis.line = element_blank())+
  theme(axis.text = element_blank()) +
  theme(axis.ticks = element_blank())+
  labs(x = NULL, y = NULL, fill = NULL)+
  ylab("End of Treatment")

}
####################################################
##Family Level Pie Chart - Group Y
##For Day-1,Day-2,Day-5 


p%>%
  select(-c(1)) %>%
  select(-c(3)) %>%
  filter(p$Group == 'Y' & p$Timepoint == 'D1') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp1
names(temp1)[2]<-paste("D1")
temp1 %>%
  mutate(D1=round(D1,digits = 2)) -> temp1

p%>%
  select(-c(1))%>%
  select(-c(3))%>%
  filter(p$Group == 'Y' & p$Timepoint == 'D2') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp2
names(temp2)[2]<-paste("D2")
temp2 %>%
  mutate(D2=round(D2,digits = 2)) -> temp2


p%>%
  select(-c(1))%>%
  select(-c(3))%>%
  filter(p$Group == 'Y' & p$Timepoint == 'D5') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp5
names(temp5)[2]<-paste("D5")
temp5 %>%
  mutate(D5=round(D5,digits = 2)) -> temp5

inner_join(temp1,temp2,by=c('Dataset')) %>%
  inner_join(temp5,by=c('Dataset')) -> InitialDays

InitialDays[is.na(InitialDays)] <- 0  

InitialDays %>%
  mutate(Values = round((D1+D2+D5)/3,digits = 2)) -> yInitialDays

yInitialDays %>%
  select(Dataset,Values) %>%
  filter(Values!=0) %>%
  mutate(Values=round(as.numeric(Values),digits=2)) %>%
  mutate(per=round(Values/sum(Values)*100,digits = 2)) %>%
  arrange(desc(per)) -> yInitialDays

yInitialDays <- rbind(yInitialDays,c(NA,colSums(yInitialDays[c(11:64),sapply(yInitialDays,is.numeric)])))
yInitialDays$Dataset[65] <- "Others"
yInitialDays <- yInitialDays[c(1:10,65),]



taxalen <- yInitialDays$Dataset[c(which(!(yInitialDays$Dataset %in% ColAssgn$taxa)))]

for (count in 1:length(taxalen)){
  if(length(taxalen) > 0){
    len = len +1
    ColAssgn %>%
      rbind(c(taxalen[count],allcols[len])) -> ColAssgn
    
  }
}



yinidays <- function(){
  yInitialDays%>%
    ggplot(aes(x='',y=reorder(Values,Values),fill = Dataset))+
    geom_bar(width=1,stat="identity")+
    scale_fill_manual(breaks = ColAssgn$taxa,
                      values = ColAssgn$cols)+
    theme_void()+
    theme_classic() +
    theme(legend.position = "top") +
    coord_polar("y",start=0) +
    theme(axis.line = element_blank())+
    theme(axis.text = element_blank()) +
    theme(axis.ticks = element_blank())+
    labs(x = NULL, y = NULL, fill = NULL)+
    ylab("Baseline")
  
}
###########################################################
##For Day-7,Day-10,Day-15

p%>%
  select(-c(1)) %>%
  select(-c(3)) %>%
  filter(p$Group == 'Y' & p$Timepoint == 'D7') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp7
names(temp7)[2]<-paste("D7")
temp7 %>%
  mutate(D7=round(D7,digits = 2)) -> temp7

p%>%
  select(-c(1))%>%
  select(-c(3))%>%
  filter(p$Group == 'Y' & p$Timepoint == 'D10') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp10
names(temp10)[2]<-paste("D10")
temp10 %>%
  mutate(D10=round(D10,digits = 2)) -> temp10


p%>%
  select(-c(1))%>%
  select(-c(3))%>%
  filter(p$Group == 'Y' & p$Timepoint == 'D15') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp15
names(temp15)[2]<-paste("D15")
temp15 %>%
  mutate(D15=round(D15,digits = 2)) -> temp15

inner_join(temp7,temp10,by=c('Dataset')) %>%
  inner_join(temp15,by=c('Dataset')) -> MidDays

MidDays[is.na(MidDays)] <- 0  

MidDays %>%
  mutate(Values = round((D7+D10+D15)/3,digits = 2)) -> yMidDays

yMidDays %>%
  select(Dataset,Values) %>%
  filter(Values!=0) %>%
  mutate(Values=round(as.numeric(Values),digits=2)) %>%
  mutate(per=round(Values/sum(Values)*100,digits = 2)) %>%
  arrange(desc(per)) -> yMidDays

yMidDays <- rbind(yMidDays,c(NA,colSums(yMidDays[c(11:59),sapply(yMidDays,is.numeric)])))
yMidDays$Dataset[60] <- "Others"        
yMidDays <- yMidDays[c(1:10,60),]


taxalen <- yMidDays$Dataset[c(which(!(yMidDays$Dataset %in% ColAssgn$taxa)))]

for (count in 1:length(taxalen)){
  if(length(taxalen) > 0){
    len = len +1
    ColAssgn %>%
      rbind(c(taxalen[count],allcols[len])) -> ColAssgn
    
  }
}


ymiddays <- function(){
yMidDays %>%
  ggplot(aes(x='',y=reorder(Values,Values),fill = Dataset))+
  geom_bar(width=1,stat="identity")+
  scale_fill_manual(breaks = ColAssgn$taxa,
                      values = ColAssgn$cols)+
  theme_void()+
  theme_classic() +
  theme(legend.position = "top") +
  coord_polar("y",start=0) +
  theme(axis.line = element_blank())+
  theme(axis.text = element_blank()) +
  theme(axis.ticks = element_blank())+
  labs(x = NULL, y = NULL, fill = NULL)+
  ylab("Mid Treatment")

}

########################################################
##For Day-20,Day-25,Day-30


p%>%
  select(-c(1)) %>%
  select(-c(3)) %>%
  filter(p$Group == 'Y' & p$Timepoint == 'D20') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp20
names(temp20)[2]<-paste("D20")
temp20 %>%
  mutate(D20=round(D20,digits = 2)) -> temp20

p%>%
  select(-c(1))%>%
  select(-c(3))%>%
  filter(p$Group == 'Y' & p$Timepoint == 'D25') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp25
names(temp25)[2]<-paste("D25")
temp25 %>%
  mutate(D25=round(D25,digits = 2)) -> temp25


p%>%
  select(-c(1))%>%
  select(-c(3))%>%
  filter(p$Group == 'Y' & p$Timepoint == 'D30') %>%
  select(-c(1:2)) %>%
  mutate_if(is.character, as.numeric) %>%
  colMeans() %>% data.frame() %>%
  rownames_to_column("Dataset") ->temp30
names(temp30)[2]<-paste("D30")
temp30 %>%
  mutate(D30=round(D30,digits = 2)) -> temp30

inner_join(temp20,temp25,by=c('Dataset')) %>%
  inner_join(temp30,by=c('Dataset')) -> Finaldays

Finaldays[is.na(Finaldays)] <- 0  

Finaldays %>%
  mutate(Values = round((D20+D25+D30)/3,digits = 2)) -> yFinaldays


yFinaldays %>%
  select(Dataset,Values) %>%
  filter(Values!=0) %>%
  mutate(Values=round(as.numeric(Values),digits=2)) %>%
  mutate(per=round(Values/sum(Values)*100,digits = 2)) %>%
  arrange(desc(per)) -> yFinaldays


yFinaldays <- rbind(yFinaldays,c(NA,colSums(yFinaldays[c(11:54),sapply(yFinaldays,is.numeric)])))
yFinaldays$Dataset[55] <- "Others"        
yFinaldays <- yFinaldays[c(1:10,55),] 



taxalen <- yFinaldays$Dataset[c(which(!(yFinaldays$Dataset %in% ColAssgn$taxa)))]

for (count in 1:length(taxalen)){
  if(length(taxalen) > 0){
    len = len +1
    ColAssgn %>%
      rbind(c(taxalen[count],allcols[len])) -> ColAssgn
    
  }
}

yfindays <- function(){
yFinaldays %>%
  ggplot(aes(x='',y=reorder(Values,Values),fill = Dataset ))+
  geom_bar(width=1,stat="identity")+
  scale_fill_manual(breaks = ColAssgn$taxa,
                      values = ColAssgn$cols)+
  theme_void()+
  theme_classic() +
  theme(legend.position = "top") +
  coord_polar("y",start=0) +
  theme(axis.line = element_blank())+
  theme(axis.text = element_blank()) +
  theme(axis.ticks = element_blank())+
  labs(x = NULL, y = NULL, fill = NULL)+
  ylab('End of Treatment') 
}

#####################################################################
##Plots

group_x <- plot_grid(
  xinidays() + theme(legend.position="none"),
  xmiddays() + theme(legend.position="none"),
  xfindays() + theme(legend.position="none"),
  nrow = 1
)

group_x_plot <-plot_grid(group_x,ncol=1
          ,axis = "tblr"
          ,labels = "Genus level comparison at multiple treatment time points for Drug X",
          label_size = 15
          )

group_y <- plot_grid(
  yinidays() + theme(legend.position="none"),
  ymiddays() + theme(legend.position="none"),
  yfindays() + theme(legend.position="none"),
  nrow = 1
)

group_y_plot <- plot_grid(group_y,ncol=1
          ,axis = "tblr"
          ,labels = "Genus level comparison at multiple treatment time points for Drug Y",
          label_size = 15
)


final_plot <- plot_grid(group_x_plot,group_y_plot,nrow = 2)	

#############################################################
#Plotting with legend
taxalegend<-as_ggplot(get_legend(ColAssgn %>%
                                   ggplot(aes(x='',y='',fill=taxa)) +
                                   geom_bar(width=1,stat="identity")+
                                   scale_fill_manual(breaks = ColAssgn$taxa,
                                                     values = ColAssgn$cols)))

                              
final_plot_with_legend <- plot_grid(plot_grid(plot_grid(group_x,ncol=1
                                                        ,axis = "tblr"
                                                        ,labels = "Genus level comparison at multiple treatment time points for Drug X",
                                                        label_size = 15,
                                                        hjust = -0.35),plot_grid(group_y,ncol=1
                                                                                 ,axis = "tblr"
                                                                                 ,labels = "Genus level comparison at multiple treatment time points for Drug Y",
                                                                                 label_size = 15,
                                                                                 hjust = -0.35),nrow = 2),taxalegend,ncol=2,rel_widths = c(0.8,0.128))


###################################################################

#ALL PLOTS

xinidays()
xmiddays()
xfindays()
yinidays()
ymiddays()
yfindays()
group_x_plot
group_y_plot
final_plot
final_plot_with_legend
