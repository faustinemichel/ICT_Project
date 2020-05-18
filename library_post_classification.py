# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:53:13 2020

@author: Faustine_michel & Gael Miramond
"""
import pandas as pd
import os
import glob

def pick_name(path,line_number,class_number):
    """
    arg :path (str of the file to read) int line number, int class_number 
    Numer of lines starts with  1
    """
    with open(path,"r") as src_file :
        for i in range(line_number):
            to_write=src_file.readline()
        src_file.close()
    trgt_file=open(str(class_number)+".txt","a")
    trgt_file.write(to_write)
    trgt_file.close()
    return
        

def supress_class_files(path,number_of_classes):
    """
    arg :path of the folder in wich class files have to be deleted, number_of_classes (ex :0->4 = 5 classes )
    ex : supress_class_files("test/",1)
    """
    list_of_files=os.listdir(path)
    list_classes=[j for j in range(number_of_classes)] #make a list with all possible filenames made by the pickname function 
    for name in list_classes:
        if(str(name)+".txt" in list_of_files)==True:
            os.remove(path+str(name)+".txt")
    return
 

def collect_results_file_to_tab(path_file_predicted_labels):
    #path_file: path where we can find the predicted labels (need to finish with a .txt)
    src_file=open(path_file_predicted_labels,"r")
    tab="" 
    tab2=[] 

    for line in src_file : 
        tab=tab+line
           
    src_file.close() # close the .obj file   

    for j in range(1, len(tab)-1, 3):
        tab2.append(int(tab[j]))

    return tab2

def results_processing(path_file_names,path_file_predicted_labels):
    tab=collect_results_file_to_tab(path_file_predicted_labels)
    
    for i in range(0,len(tab)):
        pick_name(path_file_names,i+1,tab[i])
    
    return


