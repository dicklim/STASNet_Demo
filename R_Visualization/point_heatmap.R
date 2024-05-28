#####################################Figure4 demo
library(survminer)
library(pheatmap)
library(reshape2)
library(ggplot2)
setwd('~//code_demo')
file='Demo.csv'
name=substr(file,1,12)
data=read.csv(file,row.names = 1,header = T)
colnames(data)=c('Normal','STAS','j','i','Distance')
data$tyep='Main_Tumor'
data$Distance=-data$Distance
Tumor_num=nrow(data[data$tyep=='Main_Tumor',])
if(max(data$Distance)<0){
  no_out_main_Tumor_ID=c(no_out_main_Tumor_ID,file)
}
if(max(data$Distance)>0){
  data[data$Distance>0,]$tyep='out_Main_Tumor'
}
table(data$tyep)

dis_heatmap=data.frame()
dis_heatmap[1:(max(range(data$i))-min(range(data$i))+1),1:(max(range(data$j))-min(range(data$j))+1)]=NA
rownames(dis_heatmap)=min(range(data$i)):max(range(data$i))
colnames(dis_heatmap)=min(range(data$j)):max(range(data$j))
data[data$tyep=='Main_Tumor',]$STAS=0
for(i in min(range(data$i)):max(range(data$i))){
  for(j in min(range(data$j)):max(range(data$j))){
    dis=data[data$i==i&data$j==j,]$STAS
    if(length(dis)>0){
      dis_heatmap[i+1-min(range(data$i)),j+1-min(range(data$j))]=dis
    }
    
  }
}
require(isoband)
STAS_matrix=as.matrix(dis_heatmap)
range(STAS_matrix)
STAS_matrix[is.na(STAS_matrix)]=0
bk=max(data$Distance)/min(data$Distance)

heatmap <- melt(STAS_matrix)
heatmap[,4] <- melt(distance_matrix)[,3]
heatmap$Var1=-heatmap$Var1

library(dplyr)
out_data=heatmap
colnames(out_data)=c("high","Length","STAS","Distance")
range(out_data$STAS)
out_data$type='Main_tumor'
out_data[out_data$Distance>0,]$type='Normal'
range(out_data$Distance)
table(out_data$STAS)

range(out_data$Length)
out_data=arrange(out_data,-out_data$STAS)
out_data[1:10,]$Distance=max(out_data$Distance)+100
out_data[1:10,]$STAS=out_data[1:10,]$STAS+1.01
blue='#7281B7'
red='#D39FD3'
my_palette <- c(colorRampPalette(c('#7281B7',"#F4EAEA"))(100),colorRampPalette(c("#F4EAEA",'#91CECC'))(100*abs(bk)))
my_palette=c(my_palette,'#D39FD3')
corCnaExpr=out_data
range(corCnaExpr$STAS)

ggplot(corCnaExpr, aes(x=Length,y=high)) +
  geom_point(aes(size=STAS, color=Distance)) +
  scale_color_gradientn('Distance',colors=my_palette) +
  scale_size_continuous(range = c(0,2)) + 
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, size = 8, hjust = 0.3, vjust = 0.5, color = "black"),
    axis.title = element_blank(),
    panel.border = element_rect(size = 0.7, linetype = "solid", colour = "black"),
    legend.position = "bottom",
    plot.margin = unit(c(1,1,1,1), "lines"))
ggsave(paste0('Demo.pdf'),height = nrow(m)/20,width =ncol(m)/24)
##################The output image needs to be adjusted by Ai







