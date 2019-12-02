# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:57:38 2017

@author: hfailmezger
"""
import sys
import numpy as np
import os
import pickle
import time
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
import math
from sklearn import mixture
import networkx as nx
from  rpy2.robjects import r, pandas2ri
import rpy2.robjects as robjects
from  rpy2.robjects import pandas2ri
from subprocess import call
import matplotlib.pyplot as plt

# This is a function to merge several nodes into one in a Networkx graph

def merge_nodes(G,nodes, new_node, attr_dict=None, **attr):
    """
    Merges the selected `nodes` of the graph G into one `new_node`,
    meaning that all the edges that pointed to or from one of these
    `nodes` will point to or from the `new_node`.
    attr_dict and **attr are defined as in `G.add_node`.
    """
    
    G.add_node(new_node, distToCancer=0, classCell="cancerCluster") # Add the 'merged' node
    
    for n1,n2,data in G.edges(data=True):
        # For all edges related to one of the nodes to merge,
        # make an edge going to or coming from the `new gene`.
        if n1 in nodes:
            G.add_edge(new_node,n2,data)
        elif n2 in nodes:
            G.add_edge(n1,new_node,data)
    
    for n in nodes: # remove the merged nodes
        if(G.has_node(n)):
            G.remove_node(n)
            
###############################################################################
print("Network Features")
##
pandas2ri.activate()


inputFile=sys.argv[1]
processedDataFolder=sys.argv[2]


### Change the LymIdentifier to NaN for the clustering coefficient
LymIdentifier='l'

#
distanceThreshold=7
#
filenameSplitted=inputFile.split("/")
filenamePatient=filenameSplitted[len(filenameSplitted)-1]
#
print(inputFile)
cellPos=robjects.r['load'](inputFile)
matrixW = robjects.r['CellPos']
matrix=matrixW[matrixW["class"] !='a']
matrixIndexCell=np.arange(0,matrix.shape[0])
matrixClass=matrix["class"].as_matrix()
matrixImage=matrix["image"].as_matrix()
matrixX=matrix["x"].as_matrix()
matrixY=matrix["y"].as_matrix()
matrixLocalX=matrix["xLocal"].as_matrix()
matrixLocalY=matrix["yLocal"].as_matrix()
#
graphDegreeStromal=np.empty(matrix.shape[0])
graphDegreeStromal[:] = np.NAN
graphDegreeCancer=np.empty(matrix.shape[0])
graphDegreeCancer[:] = np.NAN
graphDegreeLym=np.empty(matrix.shape[0])
graphDegreeLym[:] = np.NAN
#
graphBetweennessStromal=np.empty(matrix.shape[0])
graphBetweennessStromal[:] = np.NAN
graphBetweennessCancer=np.empty(matrix.shape[0])
#graphBetweennessCancer=[]
graphBetweennessCancer[:] = np.NAN
graphBetweennessLym=np.empty(matrix.shape[0])
graphBetweennessLym[:] = np.NAN
#
graphClosenessStromal=np.empty(matrix.shape[0])
graphClosenessStromal[:] = np.NAN
#graphClosenessCancer=np.empty(matrix.shape[0])
graphClosenessCancer=[]
#graphClosenessCancer[:] = np.NAN
graphClosenessLym=np.empty(matrix.shape[0])
graphClosenessLym[:] = np.NAN
#
graphClusteringStromal=np.empty(matrix.shape[0])
graphClusteringStromal[:] = np.NAN
#graphClusteringCancer=np.empty(matrix.shape[0])
#graphClusteringCancer[:] = np.NAN
graphClusteringCancer=[]
graphClusteringLym=np.empty(matrix.shape[0])
graphClusteringLym[:] = np.NAN
#
#
cordsBoderCells=[]
indexBorderCells=[]
subimages=np.unique(matrix["image"])
#subimages=subimages[10:50]
allLymNodes=[]
allStromalNodes=[]
allCancerClusterNodes=[]
maxNode=1
indicesBorderCells=[]
cancerClusterGraphStromal=nx.Graph()
cancerClusterGraphStromalLym=nx.Graph()
cancerCellIndexBorderCell={}
subimageNames=[]
for subimage in subimages:
    cancerNodes=[]
    stromalNodes=[]
    lymNodes=[]
    print("-------------- "+subimage+"------------------")
    subimageNames.append(subimage)
    GTemp=nx.Graph()
    indicesSubimage=np.where(matrixImage==subimage)
    cordsSubimage=np.column_stack((matrixX[indicesSubimage[0]],matrixY[indicesSubimage[0]]))
    cordsSubimageLocal=np.column_stack((matrixLocalX[indicesSubimage[0]],matrixLocalY[indicesSubimage[0]]))
    cordsSubimageLocal=cordsSubimageLocal.astype('float')
    indexCellSubimage=matrixIndexCell[indicesSubimage[0]]
    classCellSubimage=matrixClass[indicesSubimage[0]]
    ####
    indexCellSubimageOld=indexCellSubimage
    ###########################################################################
    ###########################################################################
    borderCellsX1=np.where(cordsSubimageLocal[:,0]<50)
    borderCellsX2=np.where(cordsSubimageLocal[:,0]>=1950)
    borderCellsY1=np.where(cordsSubimageLocal[:,1]<50)
    borderCellsY2=np.where(cordsSubimageLocal[:,1]>1950)
    ###########################################################################
    indicesBorderCellsSubimage = borderCellsX1
    indicesBorderCellsSubimage = np.append(indicesBorderCellsSubimage,borderCellsX2)
    indicesBorderCellsSubimage = np.append(indicesBorderCellsSubimage,borderCellsY1)
    indicesBorderCellsSubimage = np.append(indicesBorderCellsSubimage,borderCellsY2)
    indicesBorderCellsSubimage=np.unique(indicesBorderCellsSubimage.astype(int))
    indicesBorderCells=np.append(indicesBorderCells,indexCellSubimage[indicesBorderCellsSubimage])
    borderCellSubimage=np.zeros(np.max(indexCellSubimage)+1)
    borderCellSubimage[indexCellSubimage[indicesBorderCellsSubimage]]=1
    ###########################################################################
    cordsNP=cordsSubimage
    ll=np.dstack((cordsNP[:,0].astype(float),cordsNP[:,1].astype(float)))
    tree = cKDTree(ll[0])
    pairs = tree.query_pairs(distanceThreshold, p=2)
    for rc1 in range(0,len(cordsNP)):
        GTemp.add_node(str(indexCellSubimage[rc1]),classCell=classCellSubimage[rc1],distToCancer=np.nan,borderCell=borderCellSubimage[indexCellSubimage[rc1]])
        if(classCellSubimage[rc1]=="c"):
            cancerNodes=np.append(cancerNodes,str(indexCellSubimage[rc1]))
        if(classCellSubimage[rc1]=="o"):
            stromalNodes=np.append(stromalNodes,str(indexCellSubimage[rc1]))
            allStromalNodes=np.append(allStromalNodes,str(indexCellSubimage[rc1]))
        if(classCellSubimage[rc1]==LymIdentifier):
            lymNodes=np.append(allLymNodes,str(indexCellSubimage[rc1]))
            allLymNodes=np.append(allLymNodes,str(indexCellSubimage[rc1]))
        if classCellSubimage[rc1]=="c" or classCellSubimage[rc1]=="o" or classCellSubimage[rc1]==LymIdentifier:
            cancerClusterGraphStromalLym.add_node(str(indexCellSubimage[rc1]),classCell=classCellSubimage[rc1],distToCancer=np.nan,borderCell=borderCellSubimage[indexCellSubimage[rc1]])
    for pair in pairs:
            GTemp.add_edge(str(indexCellSubimage[pair[0]]),str(indexCellSubimage[pair[1]]))
            if (classCellSubimage[pair[0]] =="o" or classCellSubimage[pair[0]] ==LymIdentifier) and (classCellSubimage[pair[1]]=="o" or classCellSubimage[pair[1]]==LymIdentifier):
                cancerClusterGraphStromalLym.add_edge(str(indexCellSubimage[pair[0]]),str(indexCellSubimage[pair[1]]))
    ###########################################################################
    stromalLymNodes=np.concatenate((stromalNodes, lymNodes), axis=0)
    GTempStromalLym=GTemp.subgraph(stromalLymNodes)
    print("betweenness")
    sampleNodes=int(GTempStromalLym.number_of_nodes()/10)
    #if(sampleNodes>100):
    #    graphBetweenness=nx.betweenness_centrality(GTempStromalLym,sampleNodes)
    #else:
    #    graphBetweenness=nx.betweenness_centrality(GTempStromalLym)
    graphBetweenness=np.zeros(np.max(indexCellSubimage)+1)
    print("closeness")
    #graphCloseness=nx.closeness_centrality(GTempStromalLym)
    graphCloseness=np.zeros(np.max(indexCellSubimage)+1)
    print("clustering")
    graphClustering=nx.clustering(GTempStromalLym)
    print("degree")
    graphDegree=nx.degree(GTempStromalLym)
    #
    for nodeIndex in graphDegree:
        nodeIndexK=int(nodeIndex)
        if (GTempStromalLym.node[nodeIndex]['classCell']=='c'):
            graphDegreeCancer[nodeIndexK]=graphDegree[nodeIndex]
            graphBetweennessCancer[nodeIndexK]=graphBetweenness[nodeIndex]
            graphClosenessCancer[nodeIndexK]=graphCloseness[nodeIndexK]
            graphClusteringCancer[nodeIndexK]=graphClustering[nodeIndexK]
        elif (GTempStromalLym.node[nodeIndex]['classCell']==LymIdentifier): 
            graphDegreeLym[nodeIndexK]=graphDegree[nodeIndex]
            graphBetweennessLym[nodeIndexK]=graphBetweenness[nodeIndexK]
            graphClosenessLym[nodeIndexK]=graphCloseness[nodeIndexK]
            graphClusteringLym[nodeIndexK]=graphClustering[nodeIndex]
        elif (GTempStromalLym.node[nodeIndex]['classCell']=='o'):
            graphDegreeStromal[nodeIndexK]=graphDegree[nodeIndex]
            graphBetweennessStromal[nodeIndexK]=graphBetweenness[nodeIndexK]
            graphClosenessStromal[nodeIndexK]=graphCloseness[nodeIndexK]
            graphClusteringStromal[nodeIndexK]=graphClustering[nodeIndex]
    ###########################################################################
    cancerGraph=GTemp.subgraph(cancerNodes)
    connectedComponentsCancer=nx.connected_components(cancerGraph)
    connectedComponentsCancer=sorted(connectedComponentsCancer, key = len, reverse=True)
    for cc in range(len(connectedComponentsCancer)):
        if len(connectedComponentsCancer[cc])>50:
            clusterNodeName="cc_"+str(cc)
            cancerClusterGraphStromalLym.add_node(clusterNodeName,distToCancer=0,classCell='cancerCluster')
            allCancerClusterNodes=np.append(allCancerClusterNodes,clusterNodeName)
            for nodeInCluster in connectedComponentsCancer[cc]:
                edges=GTemp.edges(nodeInCluster)
                cancerCellIndexBorderCell[nodeInCluster]=clusterNodeName
                for edge in edges:
                    if(GTemp.node[edge[1]]['classCell']=='o' or GTemp.node[edge[1]]['classCell']==LymIdentifier):
                        cancerClusterGraphStromalLym.add_edge(clusterNodeName,edge[1])
                    if(borderCellSubimage[int(edge[1])]==1):
                            cancerCellIndexBorderCell[int(edge[1])]=clusterNodeName
###############################################################################
#############################################remove all cancer nodes###########
for node in cancerClusterGraphStromalLym.nodes():
    if cancerClusterGraphStromalLym.node[node]['classCell']   =='c':
        cancerClusterGraphStromalLym.remove_node(node)
###############################################################################
#cancerClusterGraph.remove_nodes_from(nx.isolates(cancerClusterGraph))

indicesBorderCells=indicesBorderCells.astype(int)
###############################################################################
borderGraph=nx.Graph()
###############################################################################
cordsBorder=np.column_stack((matrix["x"][indicesBorderCells],matrix["y"][indicesBorderCells]))
classBorder=matrixClass[indicesBorderCells]
#
indexCellBorder=matrixIndexCell[indicesBorderCells]
ll=np.dstack((cordsBorder[:,0].astype(float),cordsBorder[:,1].astype(float)))
tree = cKDTree(ll[0])
pairsBorder = tree.query_pairs(distanceThreshold, p=2)
for pair in pairsBorder:
      if(classBorder[pair[0]]=='o' or classBorder[pair[0]]==LymIdentifier) and (classBorder[pair[1]]=='o' or classBorder[pair[1]]==LymIdentifier ):
              cancerClusterGraphStromalLym.add_edge(str(indexCellBorder[pair[0]]),str(indexCellBorder[pair[1]]))
      if classBorder[pair[0]]=='c' and (classBorder[pair[1]]=='o' or classBorder[pair[1]]==LymIdentifier):
          if(indexCellBorder[pair[0]] in cancerCellIndexBorderCell):
              cancerCluster1=cancerCellIndexBorderCell[str(indexCellBorder[pair[0]])]
              cancerClusterGraphStromalLym.add_edge(cancerCluster1,str(indexCellBorder[pair[1]]))
      if((classBorder[pair[0]]=='o' or classBorder[pair[0]]==LymIdentifier) and classBorder[pair[1]]=='c'):
          if(indexCellBorder[pair[1]] in cancerCellIndexBorderCell):
              cancerCluster1=cancerCellIndexBorderCell[indexCellBorder[pair[1]]]
              cancerClusterGraphStromalLym.add_edge(cancerCluster1,str(indexCellBorder[pair[0]]))
      if(classBorder[pair[0]]=='c' and classBorder[pair[1]]=='c'):
          if(indexCellBorder[pair[1]] in cancerCellIndexBorderCell and indexCellBorder[pair[0]] in cancerCellIndexBorderCell):
              cancerCluster1=cancerCellIndexBorderCell[indexCellBorder[pair[0]]]
              cancerCluster2=cancerCellIndexBorderCell[indexCellBorder[pair[1]]]
              borderGraph.add_edge(cancerCluster1,cancerCluster2) 
##################### join clusters by border ################################
borderGraphConnectedComponents=nx.connected_components(borderGraph)
borderGraphConnectedComponents=sorted(borderGraphConnectedComponents, key = len, reverse=True)
bcCounter=1
for bc in range(len(borderGraphConnectedComponents)):
    nodesInComponent=[]
    for nodeInCluster in borderGraphConnectedComponents[bc]:
        nodesInComponent.append(nodeInCluster)
    if(len(nodesInComponent)>1):
        bcLabel="bc_"+str(bcCounter)
        merge_nodes(cancerClusterGraphStromal,nodesInComponent, bcLabel)
        merge_nodes(cancerClusterGraphStromalLym,nodesInComponent, bcLabel)
        bcCounter=bcCounter+1
        allCancerClusterNodes=np.append(allCancerClusterNodes,bcLabel)
cancerClusterNames = []
for (p, d) in cancerClusterGraphStromalLym.nodes(data=True):
    if d['classCell'] == 'cancerCluster':
        cancerClusterNames.append(p)
         
#
stromalCancerNodes=np.concatenate((allStromalNodes, allCancerClusterNodes), axis=0)
cancerClusterGraphStromal=cancerClusterGraphStromalLym.subgraph(stromalCancerNodes)               
#cancerClusterNames=np.unique(list(cancerCellIndexBorderCell.values()))

###############################################################################
cancerClusterGraphStromalLC = max(nx.connected_component_subgraphs(cancerClusterGraphStromal), key=len)

#cancerClusterGraphStromalLC=cancerClusterGraphStromal
#cancerClusterGraphStromalLC.remove_nodes_from(nx.isolates(cancerClusterGraphStromal))
#
cancerClusterGraphStromalLymLC = max(nx.connected_component_subgraphs(cancerClusterGraphStromalLym), key=len)
#cancerClusterGraphStromalLymLC.remove_nodes_from(nx.isolates(cancerClusterGraphStromalLym))
#cancerClusterGraphStromalLymLC=cancerClusterGraphStromalLym
#
#dotfile=open("R:\Processed\SKCM-Summary\\"+filenamePatient+"_Nodes.txt","w")
#dotfile.write("graph [\n directed  0\n")
#counter=1
#for node in cancerClusterGraphStromalLym.nodes():
#            dotfile.write(node+'\t'+cancerClusterGraphStromalLym.node[node]['classCell'] +'\n')
#dotfile.close()
counter=1
#####
#dotfile=open("R:\Processed\SKCM-Summary\\"+filenamePatient+"_grid.gml","w")
#dotfile.write("graph [\n directed  0\n")
#counter=1
#for node in cancerClusterGraphStromalLym.nodes():
#        if cancerClusterGraphStromalLym.node[node]['classCell']   =='cancerCluster':
#             dotfile.write('node [\n id "'+str(node)+ '"\n label ""\n graphics\n [ \n w 100 \n h 100 \n type "circle" \n fill "#00ff00" \n]\n]\n')
#        elif cancerClusterGraphStromalLym.node[node]['classCell']   =='cancerClusterMerged':
#             dotfile.write('node [\n id "'+str(node)+ '"\n label ""\n graphics\n [ \n w 10 \n h 10 \n type "circle" \n fill "#556b2f" \n]\n]\n')
#        elif cancerClusterGraphStromalLym.node[node]['classCell']   ==LymIdentifier:
#             dotfile.write('node [\n id "'+str(node)+ '"\n label ""\n graphics\n [\n w 10 \n h 10 \n type "circle" \n fill "#0000ff" \n]\n]\n')
#        else:
#              dotfile.write('node [\n id "'+str(node)+ '"\n label ""\n graphics\n [\n w 10 \n h 10 \n type "circle" \n fill "#ff0000" \n]\n]\n')
#for edge in cancerClusterGraphStromalLym.edges():
#    dotfile.write('edge \n [\n source "'+str(edge[0])+'"\n target "'+str(edge[1])+'"\n]\n')
#dotfile.write("]\n")
#dotfile.close()
counter=1
###############################################################################
#Extract Cells that neighbor a cancer cluster
# count stromal/Lym that neighbor a cancer cluster
# calculate node degree/clusteringcoefficient/NN

###############################################################################
pathLengths=[]
for cnI1 in range(0,len(cancerClusterNames)):
    for cnI2 in range(cnI1,len(cancerClusterNames)):
        if(cancerClusterGraphStromalLymLC.has_node(cancerClusterNames[cnI1]) and cancerClusterGraphStromalLymLC.has_node(cancerClusterNames[cnI2])):
            pathLengths.append(nx.shortest_path_length(cancerClusterGraphStromalLymLC,source=cancerClusterNames[cnI1],target=cancerClusterNames[cnI2]))
avgPathLengthCC=np.mean(pathLengths)
###############################################################################
shortestPathsLymCancer=[]
allPaths=[]
counter=0
#add Lym to stromal
#calculate path
#deleteLym
for nodeL in cancerClusterGraphStromalLymLC.nodes_iter(data=True):
    counter=counter+1
    print(counter)
    if(nodeL[1]["classCell"]=='l'):
        shortestPath=np.NAN
        allPaths=[]
        neighbors=cancerClusterGraphStromalLymLC[nodeL[0]]
        cancerClusterFound=False
        for neighbor,v in neighbors.items():
            if(cancerClusterGraphStromalLymLC.node[neighbor]["classCell"]=='cancerCluster'):
                cancerClusterFound=True
                shortestPathsLymCancer.append(1)
        if(not cancerClusterFound):
            print("cancerClusterFound")
            start=0
            for cnI1 in range(0,len(cancerClusterNames)):
                if(cancerClusterGraphStromalLC.has_node(cancerClusterNames[cnI1])):

                    if start==0:
                        lymNode=nodeL[0]
                        edgesLymNode=cancerClusterGraphStromalLym[lymNode]
                        cancerClusterGraphStromalLC.add_node(lymNode,classCell="tempLymNode",distToCancer=np.nan,borderCell=borderCellSubimage[indexCellSubimage[rc1]])
                        for edges in edgesLymNode:
                            if(cancerClusterGraphStromalLymLC.node[edges]["classCell"]=='o' and cancerClusterGraphStromalLC.has_node(edges)):
                                cancerClusterGraphStromalLC.add_edge(lymNode,edges)
                        try:
                           # break
                            shortestPath=nx.shortest_path_length(cancerClusterGraphStromalLC,source=lymNode,target=cancerClusterNames[cnI1])
                            allPaths.append(shortestPath)
                            start=start+1
                        except nx.NetworkXNoPath:
                            print('No path')
                        cancerClusterGraphStromalLC.remove_node(lymNode)
                    else:
                        lymNode=nodeL[0]
                        edgesLymNode=cancerClusterGraphStromalLym[lymNode]
                        cancerClusterGraphStromalLC.add_node(lymNode,classCell="tempLymNode",distToCancer=np.nan,borderCell=borderCellSubimage[indexCellSubimage[rc1]])
                        for edges in edgesLymNode:
                            if(cancerClusterGraphStromalLymLC.node[edges]["classCell"]=='o' and cancerClusterGraphStromalLC.has_node(edges)):
                                cancerClusterGraphStromalLC.add_edge(lymNode,edges)
                        try:
                            pathLengthLym=nx.shortest_path_length(cancerClusterGraphStromalLC,source=lymNode,target=cancerClusterNames[cnI1])
                            allPaths.append(pathLengthLym)
                            if(pathLengthLym<shortestPath):
                              shortestPath= pathLengthLym
                        except nx.NetworkXNoPath:
                            print('No path')
                        cancerClusterGraphStromalLC.remove_node(lymNode)
                    if(shortestPath==2):
                        shortestPathsLymCancer.append(shortestPath)  
                        break
           # break       
            shortestPathsLymCancer.append(shortestPath) 
shortestPathsLymCancer=np.array(shortestPathsLymCancer)
avgShortestPathsLymCancer=np.mean(shortestPathsLymCancer[~np.isnan(shortestPathsLymCancer)])
#
shortestPathsLymCancer=shortestPathsLymCancer[~np.isnan(shortestPathsLymCancer)]
#
avgShortestPathsLymCancerDistant=np.mean(shortestPathsLymCancer[np.where(shortestPathsLymCancer>1)])

numDistantLyms=len(np.where(shortestPathsLymCancer>1)[0])
numNeighborLyms=len(np.where(shortestPathsLymCancer==1)[0])
if(numNeighborLyms>0):
    ratioNeighborLyms=numDistantLyms/(numNeighborLyms+numDistantLyms)
    ratioLymAttackTrapped=numDistantLyms/numNeighborLyms
else:
    ratioNeighborLyms=np.NAN
    ratioLymAttackTrapped=np.NAN  
###############################################################################
StromalNStromal=[]
StromalNLym=[]
StromalNCancer=[]
CancerNStromal=[]
CancerNLym=[]
CancerNCancer=[]
LymNLym=[]
LymNStromal=[]
LymNCancer=[]
###############################################################################
StromalNStromalBorder=[]
StromalNLymBorder=[]
StromalNCancerBorder=[]
LymNLymBorder=[]
LymNStromalBorder=[]
LymNCancerBorder=[]
###############################################################################
StromalNStromalNonBorder=[]
StromalNLymNonBorder=[]
StromalNCancerNonBorder=[]
LymNLymNonBorder=[]
LymNStromalNonBorder=[]
LymNCancerNonBorder=[]
###############################################################################
degreeStromalNonBorder=[]
degreeLymNonBorder=[]
betweennessStromalNonBorder=[]
betweennessLymNonBorder=[]
closenessStromalNonBorder=[]
closenessLymNonBorder=[]
clusteringStromalNonBorder=[]
clusteringLymNonBorder=[]
ratioLymNonBorder=[]
####
degreeStromalBorder=[]
degreeLymBorder=[]
betweennessStromalBorder=[]
betweennessLymBorder=[]
closenessStromalBorder=[]
closenessLymBorder=[]
clusteringStromalBorder=[]
clusteringLymBorder=[]
ratioLymBorder=[]
###############################################################################
for nodeLC in cancerClusterGraphStromalLymLC.nodes_iter(data=True):
    neighbors=cancerClusterGraphStromalLymLC[nodeLC[0]]
    neighborCancer=0
    neighborStromal=0
    neighborLym=0
    cancerClusterFound=False
    for neighbor,v in neighbors.items():
        if(cancerClusterGraphStromalLymLC.node[neighbor]["classCell"]=='cancerCluster'):
            neighborCancer=neighborCancer+1
            cancerClusterFound=True
        if(cancerClusterGraphStromalLymLC.node[neighbor]["classCell"]=='o'):
            neighborStromal=neighborStromal+1
        if(cancerClusterGraphStromalLymLC.node[neighbor]["classCell"]==LymIdentifier):
            neighborLym=neighborLym+1
    if cancerClusterFound==False:
            if(cancerClusterGraphStromalLymLC.node[nodeLC[0]]["classCell"]=='o'):
                degreeStromalNonBorder.append(graphDegreeStromal[int(nodeLC[0])])
                betweennessStromalNonBorder.append(graphBetweennessStromal[int(nodeLC[0])])
                closenessStromalNonBorder.append(graphClosenessStromal[int(nodeLC[0])])
                clusteringStromalNonBorder.append(graphClusteringStromal[int(nodeLC[0])])
            if(cancerClusterGraphStromalLymLC.node[nodeLC[0]]["classCell"]==LymIdentifier):
                degreeLymNonBorder.append(graphDegreeLym[int(nodeLC[0])])
                betweennessLymNonBorder.append(graphBetweennessLym[int(nodeLC[0])])
                closenessLymNonBorder.append(graphClosenessLym[int(nodeLC[0])])
                clusteringLymNonBorder.append(graphClusteringLym[int(nodeLC[0])])
    else:
        if(cancerClusterGraphStromalLymLC.node[nodeLC[0]]["classCell"]=='o'):
                degreeStromalBorder.append(graphDegreeStromal[int(nodeLC[0])])
                betweennessStromalBorder.append(graphBetweennessStromal[int(nodeLC[0])])
                closenessStromalBorder.append(graphClosenessStromal[int(nodeLC[0])])
                clusteringStromalBorder.append(graphClusteringStromal[int(nodeLC[0])])
        if(cancerClusterGraphStromalLymLC.node[nodeLC[0]]["classCell"]==LymIdentifier):
                degreeLymBorder.append(graphDegreeLym[int(nodeLC[0])])
                betweennessLymBorder.append(graphBetweennessLym[int(nodeLC[0])])
                closenessLymBorder.append(graphClosenessLym[int(nodeLC[0])])
                clusteringLymBorder.append(graphClusteringLym[int(nodeLC[0])])
    if(cancerClusterGraphStromalLymLC.node[nodeLC[0]]["classCell"]=="cancerCluster"):
         CancerNStromal.append(neighborStromal)
         CancerNLym.append(neighborLym)
         CancerNCancer.append(neighborCancer)            
    if(cancerClusterGraphStromalLymLC.node[nodeLC[0]]["classCell"]==LymIdentifier):
         LymNStromal.append(neighborStromal)
         LymNLym.append(neighborLym)
         LymNCancer.append(neighborCancer)
         if cancerClusterFound==True: #BorderCell==True
              LymNStromalBorder.append(neighborStromal)
              LymNLymBorder.append(neighborLym)
              LymNCancerBorder.append(neighborCancer)
         else:
              LymNStromalNonBorder.append(neighborStromal)
              LymNLymNonBorder.append(neighborLym)
              LymNCancerNonBorder.append(neighborCancer)           
    if(cancerClusterGraphStromalLymLC.node[nodeLC[0]]["classCell"]=="o"):
         StromalNStromal.append(neighborStromal)
         StromalNLym.append(neighborLym)
         StromalNCancer.append(neighborCancer)
         if cancerClusterFound==True: #BorderCell==True
                 StromalNStromalBorder.append(neighborStromal)
                 StromalNLymBorder.append(neighborLym)
                 StromalNCancerBorder.append(neighborCancer)
         else:
                 StromalNStromalNonBorder.append(neighborStromal)
                 StromalNLymNonBorder.append(neighborLym)
                 StromalNCancerNonBorder.append(neighborCancer)
#################################SEARCH FOR Border Neighbor Degree ##############################################
for nodeLC in cancerClusterGraphStromalLymLC.nodes_iter(data=True):
    neighbors=cancerClusterGraphStromalLymLC[nodeLC[0]]
    cancerClusterFound=False
    for neighbor,v in neighbors.items():
        if(cancerClusterGraphStromalLymLC.node[neighbor]["classCell"]=='cancerCluster'):
            cancerClusterGraphStromalLymLC.node[nodeLC[0]]["distToCancer"]=1
            break
for nodeLC in cancerClusterGraphStromalLymLC.nodes_iter(data=True):
    if(cancerClusterGraphStromalLymLC.node[nodeLC[0]]["distToCancer"] != 1):
        neighbors=cancerClusterGraphStromalLym[nodeLC[0]]
        for neighbor,v in neighbors.items():
            if(cancerClusterGraphStromalLymLC.node[neighbor]["distToCancer"]==1):
                cancerClusterGraphStromalLymLC.node[nodeLC[0]]["distToCancer"]=2
                break      

for nodeLC in cancerClusterGraphStromalLymLC.nodes_iter(data=True):
    if(cancerClusterGraphStromalLymLC.node[nodeLC[0]]["distToCancer"] != 1 and cancerClusterGraphStromalLymLC.node[nodeLC[0]]["distToCancer"] != 2):
        neighbors=cancerClusterGraphStromalLymLC[nodeLC[0]]
        for neighbor,v in neighbors.items():
            if(cancerClusterGraphStromalLymLC.node[neighbor]["distToCancer"]==2):
                cancerClusterGraphStromalLymLC.node[nodeLC[0]]["distToCancer"]=3
                break 
###############################################################################
degreeStromalSmallDistanceCancer=[]
degreeLymSmallDistanceCancer=[]
betweennessStromalSmallDistanceCancer=[]
betweennessLymSmallDistanceCancer=[]
closenessStromalSmallDistanceCancer=[]
closenessLymSmallDistanceCancer=[]
clusteringStromalSmallDistanceCancer=[]
clusteringLymSmallDistanceCancer=[]
ratioLymSmallDistanceCancer=[]
###############################################################################
degreeStromalLargeDistanceCancer=[]
degreeLymLargeDistanceCancer=[]
betweennessStromalLargeDistanceCancer=[]
betweennessLymLargeDistanceCancer=[]
closenessStromalLargeDistanceCancer=[]
closenessLymLargeDistanceCancer=[]
clusteringStromalLargeDistanceCancer=[]
clusteringLymLargeDistanceCancer=[]
ratioLymLargeDistanceCancer=[]
###############################################################################
StromalNStromalLargeDistanceCancer=[]
StromalNLymLargeDistanceCancer=[]
StromalNCancerLargeDistanceCancer=[]
LymNLymLargeDistanceCancer=[]
LymNStromalLargeDistanceCancer=[]
LymNCancerLargeDistanceCancer=[]
###############################################################################
StromalNStromalSmallDistanceCancer=[]
StromalNLymSmallDistanceCancer=[]
StromalNCancerSmallDistanceCancer=[]
LymNLymSmallDistanceCancer=[]
LymNStromalSmallDistanceCancer=[]
LymNCancerSmallDistanceCancer=[]
###############################################################################
for nodeLC in cancerClusterGraphStromalLymLC.nodes_iter(data=True):
    neighbors=cancerClusterGraphStromalLymLC[nodeLC[0]]
    neighborCancer=0
    neighborStromal=0
    neighborLym=0
    cancerClusterFound=False
    for neighbor,v in neighbors.items():
        if(cancerClusterGraphStromalLymLC.node[neighbor]["classCell"]=='cancerCluster'):
            neighborCancer=neighborCancer+1
            cancerClusterFound=True
        if(cancerClusterGraphStromalLymLC.node[neighbor]["classCell"]=='o'):
            neighborStromal=neighborStromal+1
        if(cancerClusterGraphStromalLymLC.node[neighbor]["classCell"]==LymIdentifier):
            neighborLym=neighborLym+1
    if cancerClusterGraphStromalLymLC.node[nodeLC[0]]["distToCancer"]<=3:
            if(cancerClusterGraphStromalLymLC.node[nodeLC[0]]["classCell"]=='o'):
                degreeStromalSmallDistanceCancer.append(graphDegreeStromal[int(nodeLC[0])])
                betweennessStromalSmallDistanceCancer.append(graphBetweennessStromal[int(nodeLC[0])])
                closenessStromalSmallDistanceCancer.append(graphClosenessStromal[int(nodeLC[0])])
                clusteringStromalSmallDistanceCancer.append(graphClusteringStromal[int(nodeLC[0])])
                #
                StromalNStromalSmallDistanceCancer.append(neighborStromal)
                StromalNLymSmallDistanceCancer.append(neighborLym)
                StromalNCancerSmallDistanceCancer.append(neighborCancer)
            if(cancerClusterGraphStromalLymLC.node[nodeLC[0]]["classCell"]==LymIdentifier):
                degreeLymSmallDistanceCancer.append(graphDegreeLym[int(nodeLC[0])])
                betweennessLymSmallDistanceCancer.append(graphBetweennessLym[int(nodeLC[0])])
                closenessLymSmallDistanceCancer.append(graphClosenessLym[int(nodeLC[0])])
                clusteringLymSmallDistanceCancer.append(graphClusteringLym[int(nodeLC[0])])
                LymNStromalSmallDistanceCancer.append(neighborStromal)
                LymNLymSmallDistanceCancer.append(neighborLym)
                LymNCancerSmallDistanceCancer.append(neighborCancer)
    else:
            if(cancerClusterGraphStromalLymLC.node[nodeLC[0]]["classCell"]=='o'):
                degreeStromalLargeDistanceCancer.append(graphDegreeStromal[int(nodeLC[0])])
                betweennessStromalLargeDistanceCancer.append(graphBetweennessStromal[int(nodeLC[0])])
                closenessStromalLargeDistanceCancer.append(graphClosenessStromal[int(nodeLC[0])])
                clusteringStromalLargeDistanceCancer.append(graphClusteringStromal[int(nodeLC[0])])
                #
                StromalNStromalLargeDistanceCancer.append(neighborStromal)
                StromalNLymLargeDistanceCancer.append(neighborLym)
                StromalNCancerLargeDistanceCancer.append(neighborCancer)
            if(cancerClusterGraphStromalLymLC.node[nodeLC[0]]["classCell"]==LymIdentifier):
                degreeLymLargeDistanceCancer.append(graphDegreeLym[int(nodeLC[0])])
                betweennessLymLargeDistanceCancer.append(graphBetweennessLym[int(nodeLC[0])])
                closenessLymLargeDistanceCancer.append(graphClosenessLym[int(nodeLC[0])])
                clusteringLymLargeDistanceCancer.append(graphClusteringLym[int(nodeLC[0])])
                LymNStromalLargeDistanceCancer.append(neighborStromal)
                LymNLymLargeDistanceCancer.append(neighborLym)
                LymNCancerLargeDistanceCancer.append(neighborCancer)

###############################################################################        

avgStromalNStromal=np.mean(StromalNStromal)
avgStromalNLym=np.mean(StromalNLym)
avgStromalNCancer=np.mean(StromalNCancer)
avgCancerNStromal=np.mean(CancerNStromal)
avgCancerNLym=np.mean(CancerNLym)
avgCancerNCancer=np.mean(CancerNCancer)
avgLymNLym=np.mean(LymNLym)
avgLymNStromal=np.mean(LymNStromal)
avgLymNCancer=np.mean(LymNCancer)    
###############################################################################
#Extract Cells that neighbor a cancer cluster
# count stromal/Lym that neighbor a cancer cluster
# calculate node degree/clusteringcoefficient/NN
#############
#degreeStromalBorder=[]
#degreeLymBorder=[]
#betweennessStromalBorder=[]
#betweennessLymBorder=[]
#closenessStromalBorder=[]
#closenessLymBorder=[]
#clusteringStromalBorder=[]
#clusteringLymBorder=[]
#ratioLymBorder=[]
#################
#################
for cnI1 in range(0,len(cancerClusterNames)):
    if(cancerClusterGraphStromalLymLC.has_node(cancerClusterNames[cnI1])):
        numLym=0
        numStromal=0
        neighborStromalCC=0
        neighborLymCC=0
        neighbors=cancerClusterGraphStromalLymLC[cancerClusterNames[cnI1]]
        for neighbor,v in neighbors.items():
            ###################################################################
            neighborStromalBC=0
            neighborLymBC=0
            neighborCancerBC=0
            if(cancerClusterGraphStromalLymLC.node[neighbor]["classCell"]=='o'):
                #degreeStromalBorder.append(graphDegreeStromal[neighbor])
                #betweennessStromalBorder.append(graphBetweennessStromal[neighbor])
                #closenessStromalBorder.append(graphClosenessStromal[neighbor])
                #clusteringStromalBorder.append(graphClusteringStromal[neighbor])
                numStromal=numStromal+1
            if(cancerClusterGraphStromalLymLC.node[neighbor]["classCell"]==LymIdentifier):
                #degreeLymBorder.append(graphDegreeLym[neighbor])
                #betweennessLymBorder.append(graphBetweennessLym[neighbor])
                #closenessLymBorder.append(graphClosenessLym[neighbor])
                #clusteringLymBorder.append(graphClusteringLym[neighbor])
                numLym=numLym+1
        if ((numLym+numStromal) > 0):
            ratioLymBorder.append(numLym/(numLym+numStromal))
        else:
            ratioLymBorder.append(0)
        graphClosenessCancer.append(nx.closeness_centrality(cancerClusterGraphStromalLymLC,cancerClusterNames[cnI1]))
        #sampleNodes=int(cancerClusterGraphLC.number_of_nodes()/10)
        #if(sampleNodes>100):
        #     betweennessCancer=nx.betweenness_centrality(cancerClusterGraphLC,sampleNodes,seed=cancerClusterNames[cnI1])
        ##    betweenness_centrality_subset(betweenness_centrality_subset(G, sources=cancerClusterNames[cnI1]))
        #else:
        #        betweennessCancer=nx.betweenness_centrality(cancerClusterGraphLC,seed=cancerClusterNames[cnI1])
        #graphBetweennessCancer.append(betweennessCancer)
        graphClosenessCancer.append(nx.closeness_centrality(cancerClusterGraphStromalLymLC,cancerClusterNames[cnI1]))
        graphClusteringCancer.append(nx.clustering(cancerClusterGraphStromalLymLC,cancerClusterNames[cnI1]))
avgPathLengthCC=np.mean(pathLengths)         
###############################################################################
###
###############################################################################
graphDegreeCancer=np.array(graphDegreeCancer)
degreeCancer=np.mean(graphDegreeCancer[~np.isnan(graphDegreeCancer)])
betweennessCancer=np.mean(graphBetweennessCancer[~np.isnan(graphBetweennessCancer)])
graphClosenessCancer=np.array(graphClosenessCancer)
closenessCancer=np.mean(graphClosenessCancer[~np.isnan(graphClosenessCancer)])
graphClusteringCancer=np.array(graphClusteringCancer)
clusteringCancer=np.mean(graphClusteringCancer[~np.isnan(graphClusteringCancer)])
###
degreeLym=np.mean(graphDegreeLym[~np.isnan(graphDegreeLym)])
betweennessLym=np.mean(graphBetweennessLym[~np.isnan(graphBetweennessLym)])
closenessLym=np.mean(graphClosenessLym[~np.isnan(graphClosenessLym)])
clusteringLym=np.mean(graphClusteringLym[~np.isnan(graphClusteringLym)])
###
degreeStromal=np.mean(graphDegreeStromal[~np.isnan(graphDegreeStromal)])
betweennessStromal=np.mean(graphBetweennessStromal[~np.isnan(graphBetweennessStromal)])
closenessStromal=np.mean(graphClosenessStromal[~np.isnan(graphClosenessStromal)])
clusteringStromal=np.mean(graphClusteringStromal[~np.isnan(graphClusteringStromal)])
#####
###############################################################################
if len(LymNStromalBorder)==0:
    LymNStromalBorderAvg=np.NAN
else:
    LymNStromalBorderAvg=np.mean(LymNStromalBorder)  
if len(LymNLymBorder)==0:
    LymNLymBorderAvg=np.NAN
else:
    LymNLymBorderAvg=np.mean(LymNLymBorder)
###
if len(StromalNStromalBorder)==0:
    StromalNStromalBorderAvg=np.NAN
else:
    StromalNStromalBorderAvg=np.mean(StromalNStromalBorder)  
if len(StromalNLymBorder)==0:
    StromalNLymBorderAvg=np.NAN
else:
    StromalNLymBorderAvg=np.mean(StromalNLymBorder)
###
if len(LymNStromalNonBorder)==0:
    LymNStromalNonBorderAvg=np.NAN
else:
    LymNStromalNonBorderAvg=np.mean(LymNStromalNonBorder)  
if len(LymNLymNonBorder)==0:
    LymNLymNonBorderAvg=np.NAN
else:
    LymNLymNonBorderAvg=np.mean(LymNLymNonBorder)
###
if len(StromalNStromalNonBorder)==0:
    StromalNStromalNonBorderAvg=np.NAN
else:
    StromalNStromalNonBorderAvg=np.mean(StromalNStromalNonBorder)  
if len(StromalNLymNonBorder)==0:
    StromalNLymNonBorderAvg=np.NAN
else:
    StromalNLymNonBorderAvg=np.mean(StromalNLymNonBorder)
###
###############################################################################
if len(LymNStromalSmallDistanceCancer)==0:
    LymNStromalSmallDistanceCancerAvg=np.NAN
else:
    LymNStromalSmallDistanceCancerAvg=np.mean(LymNStromalSmallDistanceCancer)  
if len(LymNLymSmallDistanceCancer)==0:
    LymNLymSmallDistanceCancerAvg=np.NAN
else:
    LymNLymSmallDistanceCancerAvg=np.mean(LymNLymSmallDistanceCancer)
###
if len(StromalNStromalSmallDistanceCancer)==0:
    StromalNStromalSmallDistanceCancerAvg=np.NAN
else:
    StromalNStromalSmallDistanceCancerAvg=np.mean(StromalNStromalSmallDistanceCancer)  
if len(StromalNLymSmallDistanceCancer)==0:
    StromalNLymBSmallDistanceCancerAvg=np.NAN
else:
    StromalNLymSmallDistanceCancerAvg=np.mean(StromalNLymSmallDistanceCancer)
###
if len(LymNStromalLargeDistanceCancer)==0:
    LymNStromalLargeDistanceCancerAvg=np.NAN
else:
    LymNStromalLargeDistanceCancerAvg=np.mean(LymNStromalLargeDistanceCancer)  
if len(LymNLymLargeDistanceCancer)==0:
    LymNLymLargeDistanceCancerAvg=np.NAN
else:
    LymNLymLargeDistanceCancerAvg=np.mean(LymNLymLargeDistanceCancer)
###
if len(StromalNStromalLargeDistanceCancer)==0:
    StromalNStromalLargeDistanceCancerAvg=np.NAN
else:
    StromalNStromalLargeDistanceCancerAvg=np.mean(StromalNStromalLargeDistanceCancer)  
if len(StromalNLymLargeDistanceCancer)==0:
    StromalNLymLargeDistanceCancerAvg=np.NAN
else:
    StromalNLymLargeDistanceCancerAvg=np.mean(StromalNLymLargeDistanceCancer)
################################################################################
patientNameSplitted=filenamePatient.split(".")
patientNameSplitted=patientNameSplitted[0].split("-")
s = "-";
patientName=s.join([patientNameSplitted[0],patientNameSplitted[1],patientNameSplitted[2]])
#
###############################################################################                    
if(len(degreeStromalBorder)==0):
    degreeStromalBorderAvg=np.NAN
else:
    degreeStromalBorderAvg=np.mean(degreeStromalBorder)
if(len(degreeLymBorder)==0):
    degreeLymBorderAvg=np.NAN
else:
    degreeLymBorderAvg=np.mean(degreeLymBorder)
if(len(clusteringLymBorder)==0):
    clusteringLymBorderAvg=np.NAN
else:
    clusteringLymBorderAvg=np.mean(clusteringLymBorder)
if(len(betweennessLymBorder)==0):
    betweennessLymBorderAvg=np.NAN
else:
    betweennessLymBorderAvg=np.mean(betweennessLymBorder) 
if(len(betweennessStromalBorder)==0):
    betweennessStromalBorderAvg=np.NAN
else:
    betweennessStromalBorderAvg=np.mean(betweennessStromalBorder)
if(len(closenessStromalBorder)==0):
    closenessStromalBorderAvg=np.NAN
else:
    closenessStromalBorderAvg=np.mean(closenessStromalBorder)
if(len(clusteringStromalBorder)==0):
    clusteringStromalBorderAvg=np.NAN
else:
    clusteringStromalBorderAvg=np.mean(clusteringStromalBorder) 
if(len(closenessLymBorder)==0):
    closenessLymBorderAvg=np.NAN
else:
    closenessLymBorderAvg=np.mean(closenessLymBorder)
if(len(ratioLymBorder)==0):
    ratioLymBorderAvg=np.NAN
else:
    ratioLymBorderAvg=np.mean(ratioLymBorder) 
###############################################################################
if(len(degreeStromalNonBorder)==0):
    degreeStromalNonBorderAvg=np.NAN
else:
    degreeStromalNonBorderAvg=np.mean(degreeStromalNonBorder) 
if(len(degreeLymNonBorder)==0):
    degreeLymNonBorderAvg=np.NAN
else:
    degreeLymNonBorderAvg=np.mean(degreeLymNonBorder)
if(len(betweennessLymNonBorder)==0):
    betweennessLymNonBorderAvg=np.NAN
else:
    betweennessLymNonBorderAvg=np.mean(betweennessLymNonBorder)
if(len(betweennessStromalNonBorder)==0):
    betweennessStromalNonBorderAvg=np.NAN
else:
    betweennessStromalNonBorderAvg=np.mean(betweennessStromalNonBorder)
if(len(closenessStromalNonBorder)==0):
    closenessStromalNonBorderAvg=np.NAN
else:
    closenessStromalNonBorderAvg=np.mean(closenessStromalNonBorder)
if(len(closenessLymNonBorder)==0):
    closenessLymNonBorderAvg=np.NAN
else:
    closenessLymNonBorderAvg=np.mean(closenessLymNonBorder)
if(len(clusteringLymNonBorder)==0):
    clusteringLymNonBorderAvg=np.NAN
else:
    clusteringLymNonBorderAvg=np.mean(clusteringLymNonBorder)
    
if(len(clusteringStromalNonBorder)==0):
    clusteringStromalNonBorderAvg=np.NAN
else:
    clusteringStromalNonBorderAvg=np.mean(clusteringStromalNonBorder)    
###############################################################################
###############################################################################                    
if(len(degreeStromalLargeDistanceCancer)==0):
    degreeStromalLargeDistanceCancerAvg=np.NAN
else:
    degreeStromalLargeDistanceCancerAvg=np.mean(degreeStromalLargeDistanceCancer)
if(len(degreeLymLargeDistanceCancer)==0):
    degreeLymLargeDistanceCancerAvg=np.NAN
else:
    degreeLymLargeDistanceCancerAvg=np.mean(degreeLymLargeDistanceCancer)
if(len(clusteringLymLargeDistanceCancer)==0):
    clusteringLymLargeDistanceCancerAvg=np.NAN
else:
    clusteringLymLargeDistanceCancerAvg=np.mean(clusteringLymLargeDistanceCancer)
if(len(betweennessLymLargeDistanceCancer)==0):
    betweennessLymLargeDistanceCancerAvg=np.NAN
else:
    betweennessLymLargeDistanceCancerAvg=np.mean(betweennessLymLargeDistanceCancer) 
if(len(betweennessStromalLargeDistanceCancer)==0):
    betweennessStromalLargeDistanceCancerAvg=np.NAN
else:
    betweennessStromalLargeDistanceCancerAvg=np.mean(betweennessStromalLargeDistanceCancer)
if(len(closenessStromalLargeDistanceCancer)==0):
    closenessStromalLargeDistanceCancerAvg=np.NAN
else:
    closenessStromalLargeDistanceCancerAvg=np.mean(closenessStromalLargeDistanceCancer)
if(len(clusteringStromalLargeDistanceCancer)==0):
    clusteringStromalLargeDistanceCancerAvg=np.NAN
else:
    clusteringStromalLargeDistanceCancerAvg=np.mean(clusteringStromalLargeDistanceCancer) 
if(len(closenessLymLargeDistanceCancer)==0):
    closenessLymLargeDistanceCancerAvg=np.NAN
else:
    closenessLymLargeDistanceCancerAvg=np.mean(closenessLymLargeDistanceCancer)
if(len(ratioLymBorder)==0):
    ratioLymLargeDistanceCancerAvg=np.NAN
else:
    ratioLymLargeDistanceCancerAvg=np.mean(ratioLymLargeDistanceCancer) 
###############################################################################
if(len(degreeStromalNonBorder)==0):
    degreeStromalSmallDistanceCancerAvg=np.NAN
else:
    degreeStromalSmallDistanceCancerAvg=np.mean(degreeStromalSmallDistanceCancer) 
if(len(degreeLymSmallDistanceCancer)==0):
    degreeLymSmallDistanceCancerAvg=np.NAN
else:
    degreeLymSmallDistanceCancerAvg=np.mean(degreeLymSmallDistanceCancer)
if(len(betweennessLymSmallDistanceCancer)==0):
    betweennessLymSmallDistanceCancerAvg=np.NAN
else:
    betweennessLymSmallDistanceCancerAvg=np.mean(betweennessLymSmallDistanceCancer)
if(len(betweennessStromalSmallDistanceCancer)==0):
    betweennessStromalSmallDistanceCancerAvg=np.NAN
else:
    betweennessStromalSmallDistanceCancerAvg=np.mean(betweennessStromalSmallDistanceCancer)
if(len(closenessStromalSmallDistanceCancer)==0):
    closenessStromalSmallDistanceCancerAvg=np.NAN
else:
    closenessStromalSmallDistanceCancerAvg=np.mean(closenessStromalSmallDistanceCancer)
if(len(closenessLymSmallDistanceCancer)==0):
    closenessLymSmallDistanceCancerAvg=np.NAN
else:
    closenessLymSmallDistanceCancerAvg=np.mean(closenessLymSmallDistanceCancer)
if(len(clusteringLymSmallDistanceCancer)==0):
    clusteringLymSmallDistanceCancerAvg=np.NAN
else:
    clusteringLymSmallDistanceCancerAvg=np.mean(clusteringLymSmallDistanceCancer)
    
if(len(clusteringStromalSmallDistanceCancer)==0):
    clusteringStromalSmallDistanceCancerAvg=np.NAN
else:
    clusteringStromalSmallDistanceCancerAvg=np.mean(clusteringStromalSmallDistanceCancer)    
###############################################################################     
allData=[]

colnamesTable=["patientName","degreeCancer","betweennessCancer","closenessCancer","clusteringCancer","degreeLym","betweennessLym","closenessLym","clusteringLym","degreeStromal","betweennessStromal","closenessStromal","clusteringStromal","degreeStromalBorderAvg","degreeLymBorderAvg","betweennessStromalBorderAvg", "betweennessLymBorderAvg","closenessStromalBorderAvg","closenessLymBorderAvg", "clusteringStromalBorderAvg", "clusteringLymBorderAvg", "ratioLymBorder","articulationPoint","avgPathLengthCC","avgStromalNLym","avgStromalNCancer","avgStromalNStromal","avgCancerNStromal","avgCancerNLym","avgCancerNCancer","avgLymNLym","avgLymNStromal","avgLymNCancer","LymNStromalBorderAvg","LymNLymBorderAvg","StromalNStromalBorderAvg","StromalNLymBorderAvg","degreeStromalNonBorderAvg","degreeLymNonBorderAvg","betweennessLymNonBorderAvg","betweennessStromalNonBorderAvg","closenessStromalNonBorderAvg","closenessLymNonBorderAvg","LymNStromalNonBorderAvg","LymNLymNonBorderAvg","StromalNStromalNonBorderAvg","StromalNLymNonBorderAvg","clusteringStromalNonBorderAvg","avgShortestPathsLymCancer","degreeStromalSmallDistanceCancerAvg","LymNStromalSmallDistanceCancerAvg","LymNLymSmallDistanceCancerAvg","StromalNStromalSmallDistanceCancerAvg","StromalNLymSmallDistanceCancerAvg","degreeStromalLargeDistanceCancerAvg","degreeLymLargeDistanceCancerAvg","betweennessLymLargeDistanceCancerAvg","betweennessStromalLargeDistanceCancerAvg","closenessStromalLargeDistanceCancerAvg","closenessLymLargeDistanceCancerAvg","LymNStromalLargeDistanceCancerAvg","LymNLymLargeDistanceCancerAvg","StromalNStromalLargeDistanceCancerAvg","StromalNLymLargeDistanceCancerAvg","clusteringStromalLargeDistanceCancerAvg","clusteringStromalSmallDistanceCancerAvg","ratioNeighborLyms","avgShortestPathsLymCancerDistant","ratioLymAttackTrapped","numberOfSubimages"]    
allData.append([patientName,degreeCancer, betweennessCancer, closenessCancer, clusteringCancer, degreeLym, betweennessLym, closenessLym, clusteringLym,degreeStromal, betweennessStromal, closenessStromal, clusteringStromal,degreeStromalBorderAvg, degreeLymBorderAvg,betweennessStromalBorderAvg,betweennessLymBorderAvg,closenessStromalBorderAvg,closenessLymBorderAvg,clusteringStromalBorderAvg,clusteringLymBorderAvg,ratioLymBorderAvg,np.NAN,avgPathLengthCC,avgStromalNLym,avgStromalNCancer,avgStromalNStromal,avgCancerNStromal,avgCancerNLym,avgCancerNCancer,avgLymNLym,avgLymNStromal,avgLymNCancer,LymNStromalBorderAvg,LymNLymBorderAvg,StromalNStromalBorderAvg,StromalNLymBorderAvg,degreeStromalNonBorderAvg,degreeLymNonBorderAvg,betweennessLymNonBorderAvg,betweennessStromalNonBorderAvg,closenessStromalNonBorderAvg,closenessLymNonBorderAvg,LymNStromalNonBorderAvg,LymNLymNonBorderAvg,StromalNStromalNonBorderAvg,StromalNLymNonBorderAvg,clusteringStromalNonBorderAvg,avgShortestPathsLymCancer,degreeStromalSmallDistanceCancerAvg,LymNStromalSmallDistanceCancerAvg,LymNLymSmallDistanceCancerAvg,StromalNStromalSmallDistanceCancerAvg,StromalNLymSmallDistanceCancerAvg,degreeStromalLargeDistanceCancerAvg,degreeLymLargeDistanceCancerAvg,betweennessLymLargeDistanceCancerAvg,betweennessStromalLargeDistanceCancerAvg,closenessStromalLargeDistanceCancerAvg,closenessLymLargeDistanceCancerAvg,LymNStromalLargeDistanceCancerAvg,LymNLymLargeDistanceCancerAvg,StromalNStromalLargeDistanceCancerAvg,StromalNLymLargeDistanceCancerAvg,clusteringStromalLargeDistanceCancerAvg,clusteringStromalSmallDistanceCancerAvg,ratioNeighborLyms,avgShortestPathsLymCancerDistant,ratioLymAttackTrapped,len(subimages)])
###########################################################################
networkDataPatient=pd.DataFrame.from_records(allData,columns=[colnamesTable])
###
print(filenamePatient)
pickle.dump(networkDataPatient, open( os.path.join(processedDataFolder,filenamePatient+"_NetworkFeatures.p"), "wb" ) )