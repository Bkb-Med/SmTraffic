import time
import csv
import os
import math

class ExtractData:
    def __init__(self, Start_time, Fps_orig):
        self.Start_time = Start_time
        self.c_inA = {}  # car id_in:time_in    avant arrière 
        self.c_outA = {}  # car id_out:time_out   avant arrière
        self.c_inB = {}  # car id_in:time_in  gauche droit 
        self.c_outB = {}  # car id_out:time_out  gauche droit
        self.max_trackid = 0
        self.centroidx = 0
        self.centroidy = 0
        self.Fps_orig = Fps_orig
        self.rdistanceA = 3000  # rateA
        self.rdistanceB = 2500  # rateB
        self.nbr_carsA = 0
        self.mean_speedA = 0
        self.nbr_carsB = 0
        self.mean_speedB = 0
        self.cars_InOutA = {}
        self.cars_InOutB = {}
        
    def set_maxId(self, trackid):
        if trackid > self.max_trackid:
            self.max_trackid = trackid
            
    def find_centroid(self, bbox):
        self.centroidy = round((bbox[1] + bbox[3]) / 2, 2)
        self.centroidx = round((bbox[0] + bbox[2]) / 2, 2)

    def cars_in(self, time_c, trackid, bbox):
        self.find_centroid(bbox)
        if not trackid in self.c_inA and self.centroidx < 960 :
                self.c_inA[trackid] = [self.centroidx,self.centroidy,round(time_c - self.Start_time, 2)]  # car in
        if not trackid in self.c_inB and self.centroidx > 960 :
                self.c_inB[trackid] = [self.centroidx,self.centroidy,round(time_c - self.Start_time, 2)]
            
            
            
    def cars_out(self, time_c, trackid, bbox):
        self.find_centroid(bbox)
        if self.centroidx < 960 : 
            self.c_outA[trackid] = [self.centroidx,self.centroidy,round(time_c-self.Start_time, 2)]  # car out
        else :
            self.c_outB[trackid] = [self.centroidx,self.centroidy,round(time_c-self.Start_time, 2)]  # car out
            

    def find_carInOut(self, fps_stream):
        fps_coef = self.Fps_orig / fps_stream
        sumaA = 0
        sumaB = 0
        for i in range(self.max_trackid):
            if i in self.c_inA and i in self.c_outA:
                distanceInPix = math.sqrt(((self.c_outA[i][0]-self.c_inA[i][0])**2)+((self.c_outA[i][1]-self.c_inA[i][1])**2))
                print("dispx A :: "+ str(distanceInPix))
                distanceInKm = distanceInPix/self.rdistanceA
                time_frame= round(((self.c_outA[i][2]-self.c_inA[i][2])/3600), 3)
                if time_frame>0:
                    self.cars_InOutA[i] = round(
                         distanceInKm/time_frame, 2)  # en heure
            if i in self.c_inB and i in self.c_outB:
                distanceInPix = math.sqrt(((self.c_outB[i][0]-self.c_inB[i][0])**2)+((self.c_outB[i][1]-self.c_inB[i][1])**2))
                print("dispx B :: "+ str(distanceInPix))
                distanceInKm = distanceInPix/self.rdistanceB
                time_frame= round(((self.c_outB[i][2]-self.c_inB[i][2])/3600), 3)
                if time_frame>0:
                    self.cars_InOutB[i] = round(
                         distanceInKm/time_frame, 2)  # en heure
                         
        for c_v in self.cars_InOutA:
            sumaA += self.cars_InOutA[c_v]
        self.nbr_carsA = len(self.cars_InOutA)
        if self.nbr_carsA > 0:
            self.mean_speedA = round((sumaA/self.nbr_carsA)* fps_coef, 2)
       
        for c_v in self.cars_InOutB:
            sumaB += self.cars_InOutB[c_v]
        self.nbr_carsB = len(self.cars_InOutB)
        if self.nbr_carsB > 0:
            self.mean_speedB = round((sumaB/self.nbr_carsB)* fps_coef, 2)
       
        self.c_inA.clear()
        self.c_outA.clear()
        self.cars_InOutA.clear()
        self.c_inB.clear()
        self.c_outB.clear()
        self.cars_InOutB.clear()
  
        return self.nbr_carsA, self.mean_speedA,self.nbr_carsB, self.mean_speedB

