options(java.parameters = "-Xmx24g")
library(xlsx)
library(readxl)
library(hydroGOF)
library(randomForest)
library(ggplot2)
library(circlize)
library(RColorBrewer)
library(dplyr)
library(randomForestExplainer)
library(pdp)
library(tcltk)
library(ggepi)
library(patchwork)
library(caret)
library(ggrepel)
library(data.table)
library(ggraph)
library(igraph)
library(tidyverse)
library(RColorBrewer) 
library(pdp)

#####################################################################
#####                    data info (necessary)                  #####
#####################################################################
label1<-excel_sheets('~/Desktop/Genome/rf-hpo-op_Genome.xlsx')
#label2<-excel_sheets('~/Desktop/TBRFA/yfuber-TBRFA-d77db1a/burden/all-rf-op.xlsx')
label<-c(label1)
colindex<-c('MW','diameter','surface','purity','size','zp','shape', 'coating','media',
            'dissolution','illumination','duration','Nmetal','Moxygen','Xox','x','sigma_x','sigma_xnO',
            
            'transporter_activity','translation_regulator_activity','transcription_regulator_activity','catalytic_activity',
            'molecular_function_regulator','cytoskeletal_motor_activity','ATP_dependent_activity','molecular_transducer_activity',
            'molecular_adaptor_activity','antioxidant_activity','structural_molecule_activity','binding',
            
            'cellular_process','reproductive_process','localization','biological_process_involved_in_interspecies_interaction_between_organisms',
            'detoxification','reproduction','biological_regulation','response_to_stimulus','homeostatic_process','developmental_process',
            'multicellular_organismal_process','locomotion','metabolic_process','immune_system_process','biological_adhesion','signaling','biomineralization',
            
            'cytoskeletal_protein','transporter','scaffold_adaptor_protein','DNA_metabolism_protein','cell_adhesion_molecule',
            'intercellular_signal_molecule','protein_binding_activity_modulator','viral_or_transposable_element_protein',
            'RNA_metabolism_protein','gene_specific_transcriptional_regulator','defense_immunity_protein','translational_protein',
            'metabolite_interconversion_enzyme','protein_modifying_enzyme','chromatin_chromatin_binding_or_regulatory_protein','transfer_carrier_protein',
            'membrane_traffic_protein','chaperone','structural_protein','storage_protein','transmembrane_signal_receptor',
            
            'Toxicity_Conc')   #设置顺序

#####################################################################
#####                    Scatter output                         #####
#####################################################################

setwd('~/Desktop/Genome')
source(file = 'scatterplot_Genome.R',encoding = 'utf-8')

scatterlist<-list()

  scatterlist[[label[1]]]<-reg(1,1)

#scatterlist[[label2[1]]]<-reg(2,1,0,1)
#scatterlist[[label2[2]]]<-reg(2,2,0,0.4)
#scatterlist[[label2[3]]]<-reg(2,3,0,0.1)

#####################################################################
#####                    RF analysis                            #####
#####################################################################
setwd('~/Desktop/Genome')
rf.list<-list()

data<-read.xlsx('Standard_Genome.xlsx',1,header=TRUE)
  colnames(data)<-colindex

seed<-rep(1,12)
set.seed(seed[1])
rf.list[[label1[1]]]<-local({
  data=data
  randomForest(Toxicity_Conc~.,data=data,importance=TRUE,proximity=T,ntree=600,mtry=9)
})

rf<-rf.list[[label[1]]]
pre_train<-predict(rf)
cor_train<-cor(pre_train,data$Toxicity_Conc)
rmse_train<-rmse(pre_train,data$Toxicity_Conc)

print(cor_train)
save(rf.list,file='rflistGenome.rda')

#----------------------------------------------
# Importance analysis
#----------------------------------------------
# multi-way importance analysis
md<-list()
mi<-list()
colindexf<-factor(colindex[-69],levels=colindex[-69])
impframe<-data.frame(index=colindexf)

  print(1)
  rf<-rf.list[[1]]
  imp<-importance(rf)#计算变量的重要性
  incmse<-data.frame(index=names(imp[,2]),incmse=imp[,1]/max(imp[,1]))#incmse:increase in mean squared error
  colnames(incmse)[2]<-paste0(1)
  impframe<-merge(impframe,incmse,by='index',all=T)
  
  min_depth_frame<-min_depth_distribution(rf)
  md[[label[1]]]<-min_depth_frame
  
  im_frame<-measure_importance(rf)
  im_frame[4]<-im_frame[4]/max(im_frame[4])
  im_frame[5]<-im_frame[5]/max(im_frame[5])
  mi[[label[1]]]<-im_frame

#colnames(impframe)[2:20]<-label
save(impframe,md,mi,file='multi-importanceGenome.rda')

edges<-data.frame()

  edges[(1):(19),1]<-label[1]

edges[2]<-NA

for (i in 1:19){
  edges[c(seq(i,1*19,19)),3]<-colindex[i]
}

for (j in 1:19){
    edges[j,4]<-impframe[j,2]
}

edges[(1):(19),5]<-1

colnames(edges)=c('from','to','label','mse','group')

edges<-edges %>%
  group_by(group) %>%
  arrange(desc(mse), .by_group=TRUE)


for (i in 1:nrow(edges)){
  edges$to[i]<-paste0(edges$group[i],'_',edges$label[i])
}

edges_in1<-data.frame(from=c(''),to=c(''),
                      label=c(''),mse=c(0),group=c(0))
edges_in2<-data.frame(from=label1,to=label1,
                      label=label1,mse=c(rep(0,1)),group=c(rep(0,1)))

edges_all=rbind(edges_in2,edges)


nodes<-data.frame(name=c(edges_all$to),
                  label=c(edges_all$label),
                  weight=c(as.numeric(edges_all$mse)),
                  group=c(rep('B',20)))

n_leaves<-nrow(nodes)-19+1
nodes$id<-n_leaves/360*90
nodes$id[19:nrow(nodes)]<-seq(1:n_leaves)

nodes$angle<- 301.206-360 * nodes$id / n_leaves

nodes$angle[1:20]<-0

nodes$group[nodes$group==1]<-letters[1+3]


mycolor1<-c('#000000','#B2DF8A','#FB9A99',
            '#dc7bff','#a77bff','#7b84ff','#7bb8ff','#7beeff',
            '#7bffdc','#7bffa7','#84ff7b','#b8ff7b','#eeff7b',
            '#ffdc7b','#ffa77b','#ff7b84','#ff7bb8','#ff7bee')

mycolor2<-c('#000000','#B2DF8A','#FB9A99',
            '#c526ff','#6e26ff','#2635ff','#268bff','#26e2ff',
            '#26ffc5','#26ff6e','#35ff26','#8bff26','#e2ff26',
            '#ffc526','#ff6e26','#ff2635','#ff268b','#ff26e2')

mygraph <- graph_from_data_frame( edges_all, vertices = nodes, directed = TRUE )

ggraph(mygraph, layout = 'dendrogram', circular = TRUE) +
  geom_edge_diagonal(aes(colour=..index..)) +
  scale_edge_colour_distiller(palette = "RdPu") +
  geom_node_text(aes(x = x*1.25, y=y*1.25, angle = angle,label=label,color=group),
                 size=2, alpha=1) +
  geom_node_point(aes( x = x*1.07, y=y*1.07, fill=group, size=weight),
                  shape=21,stroke=0.2,color='black',alpha=1) +
  scale_colour_manual(values= mycolor2) +#??
  scale_fill_manual(values= mycolor1) +#??
  scale_size_continuous( range = c(0.5,6) ) +
  expand_limits(x = c(-1.3, 1.3), y = c(-1.3, 1.3))+
  theme_void() +
  theme(
    legend.position="none",
    plot.margin=unit(c(0,0,0,0),"cm"))+
  coord_fixed(ratio=1)

ggsave('imp_circle_r1_Genome.pdf',width=19.2,height = 5.4)

# importance plot
load(file='rflistGenome.rda')
load(file='multi-importanceGenome.rda')
mdplot<-list()
miplot<-list()

  print(1)
  min_depth_frame<-md[[1]]
  mdplot[[label[1]]]<-local({
    min_depth_frame=min_depth_frame
    plot_min_depth_distribution(min_depth_frame,k=14)+
      theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
      theme(legend.key.size = unit(1,'line'),legend.title = element_text(size=rel(1),face = "bold"),
            legend.text = element_text(size=rel(1)))
      
  })
  ggsave(paste0('md_',label[1],'.pdf'),width=7,height=7)
 
  
  plot_multi_way_importance <- function(importance_frame, x_measure = "mean_min_depth",
                                        y_measure = "times_a_root", size_measure = NULL,
                                        min_no_of_trees = 0, no_of_labels = 10,
                                        main = "Multi-way importance plot"){
    variable <- NULL
    if(any(c("randomForest", "ranger") %in% class(importance_frame))){
      importance_frame <- measure_importance(importance_frame)
    }
    data <- importance_frame[importance_frame$no_of_trees > min_no_of_trees, ]
    data_for_labels <- importance_frame[importance_frame$variable %in%
                                          important_variables(importance_frame, k = no_of_labels,
                                                              measures = c(x_measure, y_measure, size_measure)), ]
    if(!is.null(size_measure)){
      if(size_measure == "p_value"){
        data$p_value <- cut(data$p_value, breaks = c(-Inf, 0.01, 0.05, 0.1, Inf),
                            labels = c("<0.01", "[0.01, 0.05)", "[0.05, 0.1)", ">=0.1"), right = FALSE)
        plot <- ggplot(data, aes_string(x = x_measure, y = y_measure)) +
          geom_point(aes_string(color = "size_measure"), size = 3) +
          #scale_fill_manual(values =c( '#ffa77b','#dc7bff'))+
          geom_point(data = data_for_labels, color = "black", stroke = 1, aes(alpha = "top"), size = 4, shape = 21) +
          #geom_label_repel(data = data_for_labels, aes(label = variable), show.legend = FALSE,
                           #box.padding = 0.25,segment.size=0) +#label的框
          theme_bw() + scale_alpha_discrete(name = "Variable", range = c(1, 1))
      } else {
        plot <- ggplot(data, aes_string(x = x_measure, y = y_measure, size = size_measure)) +
          geom_point(aes(colour = "black")) + geom_point(data = data_for_labels, aes(colour = "blue")) +
          geom_label_repel(data = data_for_labels, aes(label = variable, size = NULL), show.legend = FALSE) +
          scale_colour_manual(name = "variable", values = c("black", "blue"), labels = c("non-top", "Top")) +
          theme_bw()
        if(size_measure == "mean_min_depth"){
          plot <- plot + scale_size(trans = "reverse")
        }
      }
    } else {
      plot <- ggplot(data, aes_string(x = x_measure, y = y_measure)) +
        geom_point(aes(colour = "black")) + geom_point(data = data_for_labels, aes(colour = "blue")) +
        geom_label_repel(data = data_for_labels, aes(label = variable, size = NULL), show.legend = FALSE) +
        scale_colour_manual(name = "variable", values = c("black", "blue"), labels = c("non-top", "Top")) +
        theme_bw()
    }
    if(x_measure %in% c("no_of_nodes", "no_of_trees", "times_a_root")){
      plot <- plot + scale_x_sqrt()
    } else if(y_measure %in% c("no_of_nodes", "no_of_trees", "times_a_root")){
      plot <- plot + scale_y_sqrt()
    }
    if(!is.null(main)){
      plot <- plot + ggtitle(main)
    }
    return(plot)
  }
  
  im_frame=mi[[1]]
  im_frame$p_value<-im_frame$p_value/5
  miplot[[label[1]]]<-local({
    im_frame=im_frame
    plot_multi_way_importance(im_frame, x_measure = "mse_increase",
                              y_measure = "node_purity_increase",
                              size_measure = "p_value", no_of_labels = 5)+
      theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
      theme(axis.line=element_line(color='black'),#坐标线颜色
            axis.ticks.length=unit(0.5,'line'))+#坐标线短线
      coord_fixed(ratio=1)+#坐标轴y/x比例
      theme(legend.position=c(0.1,0.8))
  })
  ggsave(paste0('m_im_',label[1],'.pdf'),width=5,height=5)

save(mdplot,miplot,file='importanceplotGenome.rda')
write.xlsx(im_frame,"im_frame.xlsx")
write.xlsx(impframe,"impframe.xlsx")
#----------------------------------------------
# Feature Interaction Calculate 25 most frequent interactions
#----------------------------------------------
load(file='rflistGenome.rda')
load(file='multi-importanceGenome.rda')

inter_list<-list()

  print(1)
  im_frame<-mi[[1]]
  rf<-rf.list[[1]]
  vars <- important_variables(im_frame, k = 5, measures = c("mean_min_depth","no_of_trees"))
  interactions_frame <- min_depth_interactions(rf, vars)
  interactions_frame <- arrange(interactions_frame,-interactions_frame[,4])
  inter_list[[label[1]]]<-interactions_frame
  
save(inter_list,file='interGenome.rda')

fiplot<-list()

  interactions_frame<-inter_list[[1]]
  as.vector(interactions_frame$Occurrences)
  as.vector(inter_frame$occurrences)
  hlim<-ceiling(max(interactions_frame[1:50,3],interactions_frame[1:50,6]))#
  fip<-plot_min_depth_interactions(interactions_frame,k=50)+#选取互动性最高的50个
  scale_y_continuous(limits=c(0,hlim+1.5),expand=c(0,0))+
  scale_fill_gradient(low='#0099f7', high='#f11712')+#渐变色
  theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
  theme(legend.position="bottom",legend.box="horizontal")+
  scale_y_discrete(limits=c("minimum","unconditional","occurrances"))
  fiplot[[label[1]]]<-fip
  ggsave(paste0('inter',label[1],'.pdf'),width=7.4,height=5)

save(fiplot,file='inter_plotGenome.rda')

#----------------------------------------------
# Feature Interaction Calculate
#----------------------------------------------
setwd('~/Desktop/Genome')
source('min_depth_distribution.R')
source('measure_importance.R')
source('min_depth_interactions.R')
source('interaction.R')
rdlist<-list()
r_interaction<-list()

pb <- tkProgressBar("Title","Label", 0, 100) 
for (i in 1:length(label)){
  print(paste0(i,'/',length(label)))
  info<- sprintf("?????? %d%%", round(i*100/length(label))) 
  rf<-rf.list[[i]]
  c1<-rep(1:length(colindex),each=length(colindex))
  c11<-colindex[c1]
  c2<-rep(1:length(colindex),length(colindex))
  c22<-colindex[c2]
  rd_frame<-data.frame(c11,c22)
  colnames(rd_frame)=c('variable','root_variable')
  im_frame<-mi[[i]]
  rd_frame<-merge(rd_frame,im_frame[c(1,6)],all.x=T)
  
  pb1 <- tkProgressBar("Title","Label", 0, 100) 
  for (j in 1:rf$ntree){
    info1<- sprintf("?????? %d%%", round(j*100/rf$ntree)) 
    D<-calculate_max_depth(rf,j)
    interactions_frame_single<-min_depth_interactions_single(rf,j,colindex)
    rD<-calculate_rD(D,interactions_frame_single,j)
    rD<-cbind(interactions_frame_single[1:2],rD)
    rd_frame<-merge(rd_frame,rD,by=c('variable','root_variable'),all=T)
    setTkProgressBar(pb1, j*100/rf$ntree, sprintf("???? (%s)", info1),info1) 
  }
  close(pb1)
  
  rd_frame[is.na(rd_frame)]<-0
  rdlist[[label[i]]]<-rd_frame
  for (k in 1:nrow(rd_frame)){
    rd_frame[k,504]<-sum(rd_frame[k,4:503])/rd_frame[k,3]
  }
  r_frame<-rd_frame[c(1,2,504)]
  colnames(r_frame)<-c("variable" , "root_variable" ,"r")
  r_interaction[[label[i]]]<-r_frame
  #save(rd_frame, file = paste0("Rda/rd_frame_",label[i],".rda"))
  #save(r_frame,file = paste0("Rda/r_frame_",label[i],".rda"))
  setTkProgressBar(pb, i*100/length(label), sprintf("?ܽ??? (%s)", info),info) 
}
close(pb)
save(r_interaction,file='r-interactionGenome.rda')

#----------------------------------------------
# Node and Edge files
#----------------------------------------------
setwd("~/Desktop/Genome")
load(file='r-interactionGenome.rda')
load(file='multi-importanceGenome.rda')

type<-data.frame(label=c(colindex[-69],label),
                 type=c(rep('M',69)),
                        #rep('A',6),rep('E',9),rep('y',15)),
                 color=c(rep('#98dbef',69)))
                         #rep('#a4e192',6),rep('#ffc177',9),rep('#ffb6d4',15)))

for (i in 1:length(label)){
  nodes<-data.frame(id=c(1:length(colindex)),label=c(colindex[-69],label[i]))
  nodes<-merge(nodes,type,all.x=T)
  nodes<-arrange(nodes,nodes['id'])
  #write.csv(nodes,paste0('network/nodes_',label[i],'.csv'),row.names=FALSE,fileEncoding='UTF-8')
 
  r_frame<-r_interaction[[i]]
  edges<-cbind(r_frame,c(rep('x-x',nrow(r_frame))))
  colnames(edges)<-c('Source','Target','Weight','Type')
  edges[is.na(edges)]<-0
  edges[3]<-edges[3]/max(edges[3])
  edges[3][edges[3]<0.5]<-0
  edges<-edges[-which(edges[3]==0),]
  edges<-edges[-which(edges[1]==edges[2]),]
  for (j in 1:nrow(edges)){
    j1<-which(edges[j,1]==edges[2])
    j2<-which(edges[j,2]==edges[1])
    j3<-intersect(j1,j2)
    if (length(j3)!=0){
      edges[j,3]<-mean(c(edges[j,3],edges[j3,3]))
      edges<-edges[-j3,]
   }
} 
   im_frame<-mi[[i]]
  if (i <69){
    x_y<-data.frame(Source=im_frame$variable,
                    Target=c(rep(label[i],68)),
                    Weight=im_frame[4],
                    Type=c(rep('x-y',68)))
  } 
  colnames(x_y)<-c('Source','Target','Weight','Type')
  edges<-rbind(edges,x_y)
  edges[3][edges[3]<=0]<-0
  for (j in 1:nrow(nodes[1])){
    edges[edges==nodes[j,1]]<-nodes[j,2]
  }
}
  write.csv(edges,paste0('edges.csv'),row.names=FALSE,fileEncoding='UTF-8')


#----------------------------------------------
# pdp analysis
#----------------------------------------------
pdplist<-list()
pdpplot<-list()


data<-read.xlsx('Standard_Genome.xlsx',1,header=TRUE)
colnames(data)<-colindex
train<-read.xlsx('ss-index-all_Genome_hpo.xlsx',1)$X0
    data_train<-data[train,]
    data_test<-data[-train,]
    seed<-rep(1,12)
    #seed[8]<-2
    #seed[12]<-8
    set.seed(seed[1])
    rf<-randomForest(Toxicity_Conc~.,data=data_train,importance=TRUE,proximity=T,ntree=600,mtry=9)
    
write.xlsx(interactions_frame,"interaction_frame.xlsx")
  
  inter_frame<-inter_list[[1]]
  j=1
  k=280
  subpdp<-list()
  subpdpplot<-list()
  while (j<=2){                             #选取前x个最重要的特征
    interpair<-inter_frame$interaction[k]
    v1<-strsplit(interpair,':')[[1]][1]
    v2<-strsplit(interpair,':')[[1]][2]
    k=k+1
    if (v1!=v2) {
      par<-pdp::partial(rf,pred.var = c(v1, v2), chull = TRUE, progress = "text")
      subpdp[[j]]<-par
      subpdpplot[[j]]<-autoplot(par,contour = TRUE, legend.title = "Toxicity")+
       theme_bw()+
       theme(legend.position=c(0.919,0.821),legend.background = element_rect(size=0.3,linetype = "solid",colour = "black"))+
       theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
       theme(axis.line=element_line(color='black'),
             axis.ticks.length=unit(0.5,'line'))
       ggsave(paste0('',label[1],'-',v1,'-',v2,'.pdf'),width=5,height=5)
      j<-j+1
    } else {j<-j}
  }
  pdplist[[label[1]]]<-subpdp
  pdpplot[[label[i]]]<-subpdpplot


save(pdplist,file='pdplistGenome.rda')
save(pdpplot,file='pdpplotGenome.rda')

#Useless
  subpdp<-pdplist[[1]]
  for (j in 1:4){
    par<-subpdp[[j]]
    subpdpplot[[j]]<-local({
      par=par
      ggplot(par, aes(x = par[[1L]], y = par[[2L]],
                      z = par[["yhat"]], fill = par[["yhat"]])) +
        geom_tile()+
        geom_contour(color = 'white')+
        viridis::scale_fill_viridis(name =label[i], option = 'D') +
        theme_bw()+
        xlab(colnames(par)[1])+
        ylab(colnames(par)[2])+
        theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
        theme(axis.line=element_line(color='black'),axis.text = element_text(size = rel(0.6)),
              axis.ticks.length=unit(0.3,'line'),axis.title = element_text(size = rel(0.6)))+
        theme(legend.key.size = unit(0.5,'line'),legend.title = element_text(size=rel(0.6)),
              legend.text = element_text(size=rel(0.5)),legend.position = c(0.9,0.7),
              legend.background = element_rect(size=0.5,linetype="solid",colour="black"))
      ggsave(paste0(' ',label[i],'-', colnames(par)[1],'-',colnames(par)[2],'.pdf'),width=5,height=5)
    })
  }#Useless
  pdpplot[[label[1]]]<-subpdpplot
 


  part1<-miplot[[1]]+fiplot[[1]]+plot_layout(width=c(2,4.5),height=c(2,2))+plot_annotation(tag_levels = 'a')
  part2<-pdpplot[1][1]+pdpplot[1][2]+pdpplot[1][3]+pdpplot[1][4]+
    plot_layout(ncol=2,width=c(2,2),height=c(2,2))
  part3<-mdplot[[1]]+part2+plot_layout(width=c(2.5,4.5))+
    plot_annotation(tag_levels = list(c('c','d','e','f','g')))
  
  ggsave(plot=part1,file=paste0('',label[1],'-1.pdf'),width=13,height=8)
  ggsave(plot=part3,file=paste0('',label[1],'-2.pdf'),width=13,height=7.7)


# layout <- '
# AAABBBB
# AAABBBB
# AAABBBB
# CCCDDEE
# CCCDDEE
# CCCFFGG
# CCCFFGG

setwd("~/Desktop/Genome")
datalist<-list()
      data<-read.xlsx('Standard_Genome.xlsx',1,header=TRUE)
      colnames(data)<-colindex
    datalist[[label[1]]]<-data
    
  save(datalist,file='data.rda')
ic<-c(1,7,9,3,3,7,13,14,15)
v1c=c(rep('illumination',4),rep('duration',2),rep('size',3))
v2c=c(rep('duration',4),rep('x',2),rep('diameter',3))

netpdpplot<-list()
for (j in 1:9){
  i<-ic[j]
  v1<-v1c[j]
  v2<-v2c[j]
  data<-datalist[[1]]
  rf<-randomForest(Toxicity_Conc~.,data=data,importance=TRUE,proximity=T,ntree=500,mtry=5)
  par<-pdp::partial(rf,pred.var = c(v1, v2), chull = TRUE, progress = "text")
  if (j==6) { par<-par[-which(par$L>15000),] }
  
  netpdpplot[[j]]<-local({
    par=par
    ggplot(par, aes(x = par[[1L]], y = par[[2L]],
                    z = par[["yhat"]], fill = par[["yhat"]])) +
      geom_tile()+
      geom_contour(color = 'white')+
      viridis::scale_fill_viridis(name =label[1], option = 'D') +
      theme_bw()+
      xlab(colnames(par)[1])+
      ylab(colnames(par)[2])+
      theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
      theme(axis.line=element_line(color='black'),axis.ticks.length=unit(0.3,'line'))+
      theme(legend.position = c(0.9,0.8))
    scale_x_continuous(limits = c(min(par$Zeta),60))
    ggsave(paste0(' ',label[1],'-',v1,'-',v2,'.pdf'),width=5,height=5)
  })
}


for (i in 1:9){
  netpdpplot[[i]]<-netpdpplot[[i]]+
    theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
    theme(axis.line=element_line(color='black'),axis.text = element_text(size = rel(0.6)),
          axis.ticks.length=unit(0.3,'line'),axis.title = element_text(size = rel(0.6)))+
    theme(legend.key.size = unit(0.5,'line'),legend.title = element_text(size=rel(0.6)),
          legend.text = element_text(size=rel(0.5)),legend.position = c(0.88,0.85))
}

netpdprda<-netpdpplot[[1]]+netpdpplot[[2]]+netpdpplot[[3]]+
  netpdpplot[[4]]
#+netpdpplot[[5]]+netpdpplot[[6]]+
#  netpdpplot[[7]]+netpdpplot[[8]]+netpdpplot[[9]]
 plot_layout(ncol=3,widths=rep(3,9),height=rep(3.9))
save(netpdpplot,file='netpdp.rda')
ggsave(file='figure-pdp.pdf',width=9,height=9)


#####################################################################
#####                    permutation test                       #####
#####################################################################
setwd("~/Desktop/Genome")
q2plot<-list()


  permutation<-read.xlsx('permutation_Genome.xlsx',1)

  
  lm.model<-lm(permutation$q2~permutation$r2)
  coef<-lm.model$coefficients
  q2plot[[label[1]]]<-local({
    coef=coef
    permutation=permutation
    ggplot(data=permutation,aes(r2,q2))+
      geom_point(size=2.5,color='#00BFFF')+
      geom_abline(intercept=coef[1],slope=coef[2],size=1,color='black')+
      scale_x_continuous(limits=c(0,1))+
      scale_y_continuous(limits=c(-1,1))+
      labs(title=label[1])+
      coord_fixed(ratio=2/3)+
      theme_bw()+
      theme(axis.line=element_line(color='black'),
            axis.ticks.length=unit(0.3,'line'))+
      xlab(NULL)+
      ylab(NULL)+
      theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
      annotate('text', x=0.25,y=1,label=paste0('intercept: ',round(coef[1],2)))
  })


q2p<-q2plot[[1]]

q2p<-q2p+q2plot[[1]]

q2p<-q2p+plot_layout(ncol=5)
print(q2p)
ggsave(filename = 'permutation test684.pdf',width=15,height=8.5)

#####################################################################
#####                    Correlation mat                        #####
#####################################################################
data<-read.xlsx('Standard_Genome.xlsx',1,header=TRUE)
colnames(data)<-colindex[-68]
cormat<-data.frame(cor(data))

  
  data<-read.xlsx('Standard_Genome.xlsx',1,header=TRUE)
  colnames(data)<-c(colindex[-69],label1[1])
  corsub<-cor(data)
  cormat[1:69,68]<-corsub[1:68,69]
  cormat[68,1:68]<-corsub[69,1:68]


for (i in 1:12){
  cormat[31+i,31+i]=1
}
colnames(cormat)<-c(colindex[-69],label1)
rownames(cormat)<-c(colindex[-69],label1)
write.xlsx(cormat,file = 'cormat-Genome.xlsx',col.names = T,row.names = T)

#####################################################################
#####                    Tabplot for data                       #####
#####################################################################
filelist<-dir(pattern = '.R')
for (ifile in filelist){
  source(paste0("~/Desktop/TBRFA/yfuber-TBRFA-d77db1a",ifile))
}
source('D:/?ű??ļ?/tabplot-master/build/createpalette.R')
source('D:/?ű??ļ?/bit_4.0.3/R/????????.R')
options(fftempdir = 'D:/fftemp')
#setwd('??your path??/code')
setwd("~/Desktop/TBRFA/yfuber-TBRFA-d77db1a")
library(xlsx)
#library(tabplot)
library(RColorBrewer)
library(tabplot)

data<-read.xlsx('Standard_MetOxy684.xlsx',1)
colindex<-c('MW','diameter','surface','Chemical_Purity','size','zp','Dissolution',
            'coating','illumination','duration','media_type','Species','shape','Nmetal',
            'Moxygen','Xox','x','sigma_x','sigma_xnO','Toxicity_Conc')
colnames(data)<-(colindex)
feature<-data[1:19]
#feature$Shape<-as.character(feature$Shape)
col<-colnames(feature)
#feature$Shape[which(feature$Shape=='0')]<-'0-Particles'
#feature$Shape[which(feature$Shape=='1')]<-'1-D'
#feature$Shape[which(feature$Shape=='2')]<-'2-D'
#feature$Animal[which(feature$Animal=='Mouse')]<-'Mice'
#color_enps<-c(read.xlsx('??????ɫ.xlsx',1,header=FALSE)$X1)
#col_enps<-c()
f_color<-brewer.pal(12,"Paired")
pal<-colorRampPalette(f_color)
f_color<-pal(19)
#for (i in 1:6){
#col_enps<-c(col_enps,color_enps[c(seq(i,60,by=6))])
#}
set.seed(2)
setwd("~/Desktop/TBRFA/yfuber-TBRFA-d77db1a")
tabplot::tableplot(feature,decreasing =FALSE,
                   nBins=100,
                   select=c(MW,diameter_mean,surface,Chemical_Purity,size_mean,zp_mean,Dissolution,
                            coating,illumination,duration,media_type,Species,shape,
                            Nmetal,Moxygen,Xox,x,sigma_x,sigma_xnO),
                   pals=list(Nmetal=brewer.pal(3,'Set3'),
                             Shape=brewer.pal(9,"Purples")[c(3,5,7)],
                             Functionalization=f_color))         

ggsave(filename = 'tab-imm684.pdf',width=25.2,height = 5.4)
#feature$Weight<-log10(feature$Weight)
#tabplot::tableplot(feature, decreasing=F,
                   #select=c(Types,Animal,Gender,Weight,Age),
                   #pals=list(Types=brewer.pal(8,'Set3')))

#tabplot::tableplot(feature,sortCol = Types,decreasing=F,
                   #select=c(Types,Method,Duration,Frequency,Recovery,Dose),
                   #pals=list(Types=brewer.pal(8,'Set3')))

dev.new()
set.seed(2)
tableplot(feature,sortcol=LogL,decreasing =T,
          select=c(LogL,LogD,Types,Shape,Functionalization,Zeta,SSA,
                   Animal,Gender,Weight,Age,
                   Method,Duration,Frequency,Recovery,Dose),
          pals=list(Types=brewer.pal(8,'Set3'),
                    Shape=brewer.pal(9,"Purples")[c(3,5,7)],
                    Functionalization=f_color,
                    Animal=brewer.pal(11,"Spectral")[c(4,8)],
                    Gender=brewer.pal(4,'Pastel2'),
                    Method=brewer.pal(6,'Pastel1')),
          scales='lin'
)
ggsave(filename = 'tab-imm.pdf',width=19.2,height = 25.4)


#####################################################################
#####                    Feature shuffle                        #####
#####################################################################
setwd("~/Desktop/Genome")
datalist<-list()

    data<-read.xlsx('Standard_Genome.xlsx',1,header=TRUE)
    colnames(data)<-colindex
    datalist[[label[1]]]<-data

save(datalist,file='data.rda')

seed<-rep(1,15)
seed[8]<-2
seed[12]<-8
seed[15]<-2
shuffle_list<-list()

  set.seed(seed[1])
  data<-datalist[[1]]
  rf<-randomForest(Toxicity_Conc~.,data=data,importance=TRUE,proximity=T,ntree=600,mtry=5)
  pre_train<-predict(rf)
  rmse_shuffle<-rmse(pre_train,data$Toxicity_Conc)
  
for (j in 1:(ncol(data)-1)){
    of<-data[,j]
    set.seed(j)
    sf<-sample(of,size=length(of),replace = FALSE)
    data[j]<-sf
    rf<-randomForest(Toxicity_Conc~.,data=data,importance=TRUE,proximity=T,ntree=500,mtry=5)
    pre_train<-predict(rf)
    rmse_shuffle<-append(rmse_shuffle,rmse(pre_train,data$Toxicity_Conc))
  }
shuffle_frame<-data.frame(n=c(0:(ncol(data)-1)),rmse=rmse_shuffle,increace=NA)
  for (j in 1:nrow(shuffle_frame)){
    shuffle_frame[j,3]<-(shuffle_frame[j,2]-shuffle_frame[1,2])/shuffle_frame[1,2]*100
  }
shuffle_list[[label[1]]]<-shuffle_frame

shuffleplot_r<-list()

  shuffle_frame<-shuffle_list[[1]]
  colnames(shuffle_frame)<-c('n_feature','rmse','increase')
  shuffleplot_r[[label[1]]]<-local({
    shuffle_frame=shuffle_frame
    ggplot(shuffle_frame,aes(n_feature,increase))+
      geom_line(color='#c526ff')+
      geom_point(color='#c526ff')+
      theme_bw()+
      labs(title=label[1])+
      theme(axis.line=element_line(color='black'))+
      theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
      coord_fixed(ratio=nrow(shuffle_frame)/
                    (max(shuffle_frame$increase)-min(shuffle_frame$increase)))+
      theme(legend.position="none")
  })

save(shuffleplot_r,file = 'shuffleplot_r.rda')
p<-shuffleplot_r[[1]]

p<-p+plot_layout(ncol=3)
ggsave('feature-shuffle-Genome.pdf',width = 9,height=15)
write.xlsx(shuffle_frame,"shuffle_frame.xlsx")
