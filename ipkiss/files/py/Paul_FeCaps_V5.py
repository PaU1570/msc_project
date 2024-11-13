#! python
#
#
# ----------------------------------------------------------------------------------
# import the IPKISS library
# ----------------------------------------------------------------------------------
from ipkiss.all import *
from ipkiss.process.layer_map import UnconstrainedGdsiiPPLayerInputMap, UnconstrainedGdsiiPPLayerOutputMap
# from ipkiss.process.layer_map import *
from ipkiss.primitives import font as f2

import sys
import math
import csv

TECH.METRICS.GRID = 1E-9
TECH.METRICS.UNIT = 1E-6
MYGRIDSIZE = 1E-9  # in principle, it should be set by TECH.METRICS.GRID, but it does not work in some cases for unknown reasons (bug?)

#----------------------------------------------------------------------------------
# DEFINITIONS / PARAMETERS
#----------------------------------------------------------------------------------

# filename
FILENAMEPREFIX = "Paul_FeCaps"
Version = 5
FILENAME = "%s_V%0.f.gds" % (FILENAMEPREFIX, Version)  # name of gds file
# define fine structures < 1.0 um
fineLimit = 1.0
# Processes
rePack = True


#---------------------------------- Layer definition Estoril
# layer export settings
MET_CH_1 = ProcessLayer("Channel", "MET_CH_1")
MET_SD_2 = ProcessLayer("Metal contact to channel(VO2)", "MET_SD_2")
MET_TE_3 = ProcessLayer("Top electrode (Mesa) definition", "MET_TE_3")
VIA_CL_4 = ProcessLayer("HZO opening VlA to channel contact", "VIA_CL_4")
VIA_SDG_5 = ProcessLayer("Passivation opening VlA from Ml to SD and G", "VIA_SDG_5")
MET_M1_6 = ProcessLayer("Metal Line and Pads", "MET_M1_6")
#BOOL = ProcessLayer("Boolean Operator","BOOL")
LABEL = ProcessLayer("labels", "LABEL")
FULL_LABEL = ProcessLayer("Full labels", "FULL_LABEL")

DWLMARKS = ProcessLayer("DWLMarks", "DWLMARKS")
OPTMARKS_L1 = ProcessLayer("Optical markers layer 1", "OPTMARKS_L1")
OPTMARKS_L2 = ProcessLayer("Optical markers layer 2", "OPTMARKS_L2")

# layer colors
process_layer_color_map = {MET_CH_1: '#000000', MET_TE_3: '#747474', 
                           MET_SD_2: '#FAC30D', OPTMARKS_L1: '#ffbf00', OPTMARKS_L2: '#ccccff',
                           VIA_SDG_5: '#008100', MET_M1_6: '#BFBFBF', 
                           VIA_CL_4: '#8F4B04', DWLMARKS: '#000000', 
                           LABEL: '#FF00FF', FULL_LABEL: '#FF00FF'}

# Purposes
DEP = PatternPurpose(name="Deposition", extension="DEP")
COMMENT = PatternPurpose(name="Comment", extension="COMM")
ETCH = PatternPurpose(name="Etching", extension="ETCH")
TEMP = PatternPurpose(name="temp", extension="temp")
VIEW = PatternPurpose(name="View", extension="VIEW")

# layer export settings
process_layer_map = {MET_CH_1:1, MET_SD_2: 2,  MET_TE_3: 3, VIA_SDG_5: 5,  VIA_CL_4: 4,  MET_M1_6: 6, LABEL: 30, FULL_LABEL: 29, DWLMARKS: 109, OPTMARKS_L1: 110, OPTMARKS_L2: 111} # BOOL: 31,


# datatype export settings
purpose_datatype_map = {DEP: 1, COMMENT: 99, ETCH: 2, TEMP: 500, VIEW: 4}


# define the export / import maps to and from GDS2
TECH.GDSII.EXPORT_LAYER_MAP = UnconstrainedGdsiiPPLayerOutputMap(process_layer_map=process_layer_map,
                                                                 purpose_datatype_map=purpose_datatype_map)

# globals
global strucCount
strucCount = 0

global deviceNbr
deviceNbr = 0

Ccl = 2.0  # clearance for optical
Ccl_big = 5.0  # clearance for test structures (FeCAP 100 um)

#----------------------------------------------------------------------------------
# DEFINE Functions
#----------------------------------------------------------------------------------
def writeLayoutStyleFile(process_layer_map, purpose_datatype_map):
    print(process_layer_map)
    text_file = open("%s.lyp" % (FILENAMEPREFIX), "w")
    text_file.write("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<layer-properties>")
    for process_layer in process_layer_map:
        print(process_layer)
        print(process_layer.extension)
        print(process_layer_map[process_layer])

        for purposetype in purpose_datatype_map:
            text_file.write("\n <properties>")
            text_file.write("\n  <frame-color>%s</frame-color>" % (process_layer_color_map[process_layer]))
            text_file.write("\n  <fill-color>%s</fill-color>" % (process_layer_color_map[process_layer]))
            text_file.write("\n  <frame-brightness>%i</frame-brightness>" % (0))
            text_file.write("\n  <fill-brightness>%i</fill-brightness>" % (0))
            text_file.write("\n  <dither-pattern>%s</dither-pattern>" % ('I9'))
            text_file.write("\n  <valid>%s</valid>" % ('true'))
            text_file.write("\n  <visible>%s</visible>" % ('true'))
            text_file.write("\n  <transparent>%s</transparent>" % ('false'))
            text_file.write("\n  <width/>")
            text_file.write("\n  <marked>%s</marked>" % ('false'))
            text_file.write("\n  <animation>%i</animation>" % (0))
            text_file.write("\n  <name>%s/%s</name>" % (process_layer.extension, purposetype.extension))
            text_file.write(
                "\n  <source>%i/%i@1</source>" % (process_layer_map[process_layer], purpose_datatype_map[purposetype]))
            text_file.write("\n </properties>")

    print(purpose_datatype_map)
    text_file.write("\n</layer-properties>")
    text_file.close()


def placeGDSonWafer(GDSfilePath="", process_layer_map={},
                    purpose_datatype_map={}):
    # loads a GDS file, and puts it at a certain position on the wafer

    global strucCount
    strucCount += 1
    elems = Structure(name="importedGDS_%i" % (strucCount))
    print "starting import of %s" % (GDSfilePath)
    curFilenameGds = GDSfilePath
    importedGDS = InputGdsii(file(curFilenameGds, "rb"))
    importedGDS.layer_map = UnconstrainedGdsiiPPLayerInputMap(process_layer_map=process_layer_map,
                                                              purpose_datatype_map=purpose_datatype_map)
    importedLayout = importedGDS.read()
    # everything in the DIE is top level
    topLevelStructure = importedLayout.unreferenced_structures()
    topLevelStructure[0].name = "%s_%i" % (topLevelStructure[0].name, strucCount)
    print(topLevelStructure)
    elems += SRef(topLevelStructure[0])
    return elems


class FeCAP(Structure):
    FEcontactSize = PositiveNumberProperty(required=True)  # width of the metal via contact to junction
    MOcontactSize = PositiveNumberProperty(required=True)  # width of the metal via contact to junction
    viaDistance = PositiveNumberProperty(default=5.0)  # distance defining the wedge
    islandCcl = PositiveNumberProperty(default=0.5)  # distance between the mesa and the edge of the bottom electrode
    label = StringProperty(default="lFeCAP")
    label_full = StringProperty(default="lFeCAP")
    withoutFEMesa = BoolProperty(default=False)
    #Ccl = PositiveNumberProperty(default=0.05)  # clearance (e-beam)

    def define_name(self):
        name = self.label
        return name
    
    def define_elements(self, elems):

        FEOpeningCenter = Coord2(0,0)
        Viacenter = Coord2(100.0,0)    

        # Mesa definition, define device size
        # for leakage current measurements, don't pattern a Mesa.
        if self.withoutFEMesa == False:
            o1 = Coord2(-2*self.FEcontactSize/3, self.FEcontactSize)
            o2 = Coord2(2*self.FEcontactSize/3, self.FEcontactSize)
            o3 = Coord2(self.FEcontactSize , 2*self.FEcontactSize/3)
            o4 = Coord2(self.FEcontactSize , -2*self.FEcontactSize/3)
            o5 = Coord2(2*self.FEcontactSize/3 , -self.FEcontactSize)
            o6 = Coord2(-2*self.FEcontactSize/3 , -self.FEcontactSize)
            o7 = Coord2(-self.FEcontactSize , -2*self.FEcontactSize/3 )
            o8 = Coord2(-self.FEcontactSize , 2*self.FEcontactSize/3)
            '''
            if self.FEcontactSize <= fineLimit:
                elems += Boundary(layer=ProcessPurposeLayer(process=MET_TE_3_FINE, purpose=ETCH), 
                                  shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True))
            else: '''
            elems += Boundary(layer=ProcessPurposeLayer(process=MET_TE_3, purpose=ETCH), 
                              shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True)) 

        # FE/MO island etch      
        p1 = Coord2(- self.FEcontactSize - self.islandCcl, - self.FEcontactSize - self.islandCcl)
        p2 = Coord2(self.viaDistance, - self.FEcontactSize - self.islandCcl)
        p3 = Coord2(2*self.viaDistance , (p2.y-27.5)/2.0) 
        p4 = Coord2(3*self.viaDistance , (p2.y-27.5)/2.0) 
        p5 = Coord2(55.0 , - 27.5)
        p6 = Coord2(150 , - 27.5)
        p7 = Coord2(150 , 27.5)  
        p8 = Coord2(55.0 , 27.5)
        p9 = Coord2(3*self.viaDistance , -(p2.y-27.5)/2.0) 
        p10 = Coord2(2*self.viaDistance , -(p2.y-27.5)/2.0) 
        p11 = Coord2(self.viaDistance, self.FEcontactSize + self.islandCcl)
        p12 = Coord2(- self.FEcontactSize - self.islandCcl, self.FEcontactSize + self.islandCcl)      
      
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                              shape=Shape(points=[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12], closed=True))
        
        #Generate the left pad 
        q1 = Coord2(- 2*self.viaDistance , -(p2.y-27.5)/2.0)
        q2 = Coord2(- 3*self.viaDistance , -(p2.y-27.5)/2.0)
        q3 = Coord2(- 50.0 , 50.0)
        q4 = Coord2(- 150.0 , 50.0)
        q5 = Coord2(- 150.0 , - 50.0)
        q6 = Coord2(- 50.0 , - 50.0)
        q7 = Coord2(- 3*self.viaDistance , (p2.y-27.5)/2.0) 
        q8 = Coord2(- 2*self.viaDistance , (p2.y-27.5)/2.0)
        q9 = Coord2(- self.viaDistance, - self.FEcontactSize - self.islandCcl)
        q10 = Coord2(self.FEcontactSize + self.islandCcl, - self.FEcontactSize - self.islandCcl)        
        q11 = Coord2(self.FEcontactSize + self.islandCcl, self.FEcontactSize + self.islandCcl)
        q12 = Coord2(- self.viaDistance, self.FEcontactSize + self.islandCcl)
      
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                              shape=Shape(points=[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12], closed=True))    
        
        # Structure Fill MESA + BE
        f1=Coord2(-150, 90-207.5)
        f2=Coord2(-150, p7.y+22.5)
        f3=Coord2(150, p7.y+22.5)
        f4=Coord2(150, p7.y+4)
        f5=Coord2(50, 31.5)
        f6=Coord2(3*self.viaDistance, -p4.y+4)
        f7=Coord2(2*self.viaDistance, -p3.y+4)
        f8=Coord2(self.viaDistance, -p2.y+4)
        #f9=Coord2(-self.viaDistance, -p1.y+4)
        f10=Coord2(p12.x-4,p12.y+4)
        f11=Coord2(p1.x-4,p1.y-4)
        #f10=Coord2(- 2*self.viaDistance , -q8.y+4)
        #f11=Coord2(- 2*self.viaDistance , q8.y-4)
        #f12=Coord2(-self.viaDistance, p1.y-4)
        f13=Coord2(self.viaDistance, p2.y-4)
        f14=Coord2(2*self.viaDistance, p3.y-4)
        f15=Coord2(3*self.viaDistance, p4.y-4)
        f16=Coord2(50, -31.5)
        f17=Coord2(150, p5.y-4)
        f18=Coord2(150, 90-207.5)
        
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_TE_3, purpose=ETCH),
                              shape=Shape(points=[f1, f2, f3, f4, f5, f6, f7, f8, f10, f11, f13, f14, f15, f16, f17, f18], closed=True))        
        
        w1=Coord2(f1.x - Ccl , f2.y + Ccl - 171.5)
        w2=Coord2(f2.x - Ccl , f2.y + Ccl)
        w3=Coord2(f3.x + Ccl , f3.y + Ccl)
        w4=Coord2(f4.x + Ccl , f4.y - Ccl)
        w5=Coord2(f5.x , f5.y - Ccl)
        w6=Coord2(f6.x , f6.y - Ccl)
        w7=Coord2(f7.x , f7.y - Ccl)
        w8=Coord2(f8.x , f8.y - Ccl)
        #w9=Coord2(f9.x , f9.y - Ccl)
        w10=Coord2(f10.x + Ccl , f10.y - Ccl)
        w11=Coord2(f11.x + Ccl , f11.y + Ccl)
        #w12=Coord2(f12.x , f12.y + Ccl)
        w13=Coord2(f13.x , f13.y + Ccl)
        w14=Coord2(f14.x , f14.y + Ccl)
        w15=Coord2(f15.x , f15.y + Ccl)
        w16=Coord2(f16.x , f16.y + Ccl)
        w17=Coord2(f17.x + Ccl , f17.y + Ccl)
        w18=Coord2(f18.x + Ccl , f2.y + Ccl - 171.5)
        
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                              shape=Shape(points=[w1, w2, w3, w4, w5, w6, w7, w8, w10, w11, w13, w14, w15, w16, w17, w18], closed=True))        

        j1=Coord2(45, 15)
        j2=Coord2(45,-15)
        j3=Coord2(self.FEcontactSize + 10, -2*self.FEcontactSize/3)
        j4=Coord2(self.FEcontactSize + 10, 2*self.FEcontactSize/3)
        
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_TE_3, purpose=ETCH),
                              shape=Shape(points=[j1, j2, j3, j4], closed=True))
        
              
        # Add metallic contact to bottom electrode
        elems += Rectangle(layer=ProcessPurposeLayer(process=MET_SD_2, purpose=ETCH),
                               center=Viacenter, box_size=(90.0, 50.0))        
        # Open passivation to bottom electrode
        elems += Rectangle(layer=ProcessPurposeLayer(process=VIA_CL_4, purpose=ETCH),
                               center=Viacenter, box_size=(80.0, 40.0))
        # Connect the HZO via to the top pad
        elems += Rectangle(layer=ProcessPurposeLayer(process=VIA_SDG_5, purpose=ETCH),
                               center=Viacenter, box_size=(85.0, 45.0))        
        # Metallise contact to bottom electrode:
        elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                               center=Viacenter, box_size=(100.0, 171.5))
        # create q box for Boolean operation:
        '''elems += Rectangle(layer=ProcessPurposeLayer(process=BOOL, purpose=ETCH),
                               center=FEOpeningCenter, box_size=(305.0, 145.0))'''        
         
        # Open passivation to FE mesa
        if self.withoutFEMesa == False:
                '''
                if self.FEcontactSize <= fineLimit:
                    o1 = Coord2(-2*(self.FEcontactSize-Ccl)/3, (self.FEcontactSize-Ccl))
                    o2 = Coord2(2*(self.FEcontactSize-Ccl)/3, (self.FEcontactSize-Ccl))
                    o3 = Coord2((self.FEcontactSize-Ccl) , 2*(self.FEcontactSize-Ccl)/3)
                    o4 = Coord2((self.FEcontactSize-Ccl) , -2*(self.FEcontactSize-Ccl)/3)
                    o5 = Coord2(2*(self.FEcontactSize-Ccl)/3 , -(self.FEcontactSize-Ccl))
                    o6 = Coord2(-2*(self.FEcontactSize-Ccl)/3 , -(self.FEcontactSize-Ccl))
                    o7 = Coord2(-(self.FEcontactSize-Ccl) , -2*(self.FEcontactSize-Ccl)/3 )
                    o8 = Coord2(-(self.FEcontactSize-Ccl) , 2*(self.FEcontactSize-Ccl)/3)                

                    elems += Boundary(layer=ProcessPurposeLayer(process=VIA_SDG_5_FINE, purpose=ETCH), 
                                  shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True))
                else: '''
                o1 = Coord2(-2*(self.FEcontactSize-Ccl)/3, (self.FEcontactSize-Ccl))
                o2 = Coord2(2*(self.FEcontactSize-Ccl)/3, (self.FEcontactSize-Ccl))
                o3 = Coord2((self.FEcontactSize-Ccl) , 2*(self.FEcontactSize-Ccl)/3)
                o4 = Coord2((self.FEcontactSize-Ccl) , -2*(self.FEcontactSize-Ccl)/3)
                o5 = Coord2(2*(self.FEcontactSize-Ccl)/3 , -(self.FEcontactSize-Ccl))
                o6 = Coord2(-2*(self.FEcontactSize-Ccl)/3 , -(self.FEcontactSize-Ccl))
                o7 = Coord2(-(self.FEcontactSize-Ccl) , -2*(self.FEcontactSize-Ccl)/3 )
                o8 = Coord2(-(self.FEcontactSize-Ccl) , 2*(self.FEcontactSize-Ccl)/3)                
                elems += Boundary(layer=ProcessPurposeLayer(process=VIA_SDG_5, purpose=ETCH), 
                                  shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True))
        # for leakage current measurements, open above the Mesa with layer 4
        if self.withoutFEMesa == True:
                '''
                if self.FEcontactSize <= fineLimit:
                    o1 = Coord2(-2*(self.FEcontactSize-Ccl)/3, (self.FEcontactSize-Ccl))
                    o2 = Coord2(2*(self.FEcontactSize-Ccl)/3, (self.FEcontactSize-Ccl))
                    o3 = Coord2((self.FEcontactSize-Ccl) , 2*(self.FEcontactSize-Ccl)/3)
                    o4 = Coord2((self.FEcontactSize-Ccl) , -2*(self.FEcontactSize-Ccl)/3)
                    o5 = Coord2(2*(self.FEcontactSize-Ccl)/3 , -(self.FEcontactSize-Ccl))
                    o6 = Coord2(-2*(self.FEcontactSize-Ccl)/3 , -(self.FEcontactSize-Ccl))
                    o7 = Coord2(-(self.FEcontactSize-Ccl) , -2*(self.FEcontactSize-Ccl)/3 )
                    o8 = Coord2(-(self.FEcontactSize-Ccl) , 2*(self.FEcontactSize-Ccl)/3)                

                    elems += Boundary(layer=ProcessPurposeLayer(process=VIA_CL_4_FINE, purpose=ETCH),       
                                      shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True))
                else: '''
                o1 = Coord2(-2*(self.FEcontactSize-Ccl)/3, (self.FEcontactSize-Ccl))
                o2 = Coord2(2*(self.FEcontactSize-Ccl)/3, (self.FEcontactSize-Ccl))
                o3 = Coord2((self.FEcontactSize-Ccl) , 2*(self.FEcontactSize-Ccl)/3)
                o4 = Coord2((self.FEcontactSize-Ccl) , -2*(self.FEcontactSize-Ccl)/3)
                o5 = Coord2(2*(self.FEcontactSize-Ccl)/3 , -(self.FEcontactSize-Ccl))
                o6 = Coord2(-2*(self.FEcontactSize-Ccl)/3 , -(self.FEcontactSize-Ccl))
                o7 = Coord2(-(self.FEcontactSize-Ccl) , -2*(self.FEcontactSize-Ccl)/3 )
                o8 = Coord2(-(self.FEcontactSize-Ccl) , 2*(self.FEcontactSize-Ccl)/3)                
                elems += Boundary(layer=ProcessPurposeLayer(process=VIA_CL_4, purpose=ETCH), 
                                    shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True)) 
       
        # create Label
        full_labelHeight = 24
        font1 = TEXT_FONT_STANDARD.modified_copy()
        font1.spacing = 0.00
        font1.line_width = 0.1
        label_full = PolygonText(layer=ProcessPurposeLayer(process=FULL_LABEL, purpose=COMMENT), 
                                     text=self.label_full,
                                    coordinate=Coord2(-93, 10.0), 
                                    font=font1, 
                                    height=full_labelHeight)
        elems += label_full        
        labelHeight = 60
        font1 = TEXT_FONT_STANDARD.modified_copy()
        font1.spacing = 0.00
        font1.line_width = 0.1
        label = PolygonText(layer=ProcessPurposeLayer(process=LABEL, purpose=COMMENT), 
                            text=self.label,
                            coordinate=Coord2(0, -52), 
                            font=font1, 
                            height=labelHeight)
        elems += label
        
        elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                               center=Coord2(-51, -85.75), box_size=(198, 67.5))
        
        return elems


class FourProbesMetalLines(Structure):
    l = PositiveNumberProperty(default=1000.0, doc="length of metal line")  # total lenght
    w = PositiveNumberProperty(default=2.0, doc="width of metal line")
    padSize = PositiveNumberProperty(default=80.0, doc="Size of Pad")
    padSpacing = PositiveNumberProperty(default=20.0, doc="Size of Pad")
    LineLayer = StringProperty(default="MET_M1_6")
    textLabel = StringProperty(default="Label")

    def define_elements(self, elems):

        # create pads
        centerPad1 = Coord2(self.padSize / 2.0, self.padSize / 2.0)
        centerPad2 = Coord2(centerPad1.x + self.padSize + self.padSpacing, centerPad1.y)
        centerPad3 = Coord2(centerPad2.x + self.padSize + self.padSpacing, centerPad1.y)
        centerPad4 = Coord2(centerPad3.x + self.padSize + self.padSpacing, centerPad1.y)

        if self.LineLayer == "MET_M1_6" or self.LineLayer == "MET_CH_1" :
            elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                               center=centerPad1,
                               box_size=(self.padSize, self.padSize))
            elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                               center=centerPad2,
                               box_size=(self.padSize, self.padSize))
            elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                               center=centerPad3,
                               box_size=(self.padSize, self.padSize))
            elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                               center=centerPad4,
                               box_size=(self.padSize, self.padSize))


        if self.LineLayer == "MET_CH_1" :
            elems += Rectangle(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                               center=centerPad1,
                               box_size=(self.padSize, self.padSize))
            elems += Rectangle(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                               center=centerPad2,
                               box_size=(self.padSize, self.padSize))
            elems += Rectangle(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                               center=centerPad3,
                               box_size=(self.padSize, self.padSize))
            elems += Rectangle(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                               center=centerPad4,
                               box_size=(self.padSize, self.padSize))

            elems += Rectangle(layer=ProcessPurposeLayer(process=VIA_SDG_5, purpose=ETCH),
                               center=centerPad1,
                               box_size=(self.padSize - Ccl, self.padSize - Ccl))
            elems += Rectangle(layer=ProcessPurposeLayer(process=VIA_SDG_5, purpose=ETCH),
                               center=centerPad2,
                               box_size=(self.padSize - Ccl, self.padSize - Ccl))
            elems += Rectangle(layer=ProcessPurposeLayer(process=VIA_SDG_5, purpose=ETCH),
                               center=centerPad3,
                               box_size=(self.padSize - Ccl, self.padSize - Ccl))
            elems += Rectangle(layer=ProcessPurposeLayer(process=VIA_SDG_5, purpose=ETCH),
                               center=centerPad4,
                               box_size=(self.padSize - Ccl, self.padSize - Ccl))


        # create connecting metal line

        # check if l is larger than padSpacing
        if self.l > (12 * self.padSpacing + 5 * self.padSize):

            # Long line:
            length1 = self.l / 2.0 - 3 * self.padSpacing + self.padSize / 2.0
            centerMetalLine1 = Coord2(self.padSize / 2.0 + length1 / 2.0, self.padSize + 4 * self.padSpacing)
            # Short line:
            length2 = self.l / 2.0 - 5 * self.padSpacing - 3 * self.padSize / 2.0
            centerMetalLine2 = Coord2(2.5 * self.padSize + 2.0 * self.padSpacing + length2 / 2.0,
                                      self.padSize + 2.0 * self.padSpacing)
            # Turn line:
            centerMetalLine3 = Coord2(self.l / 2.0 - 3 * self.padSpacing + self.padSize,
                                      self.padSize + 3 * self.padSpacing)
            # Connectors:
            centerMetalLineI1 = Coord2(self.padSize / 2.0, self.padSize + 2.0 * self.padSpacing)
            centerMetalLineV1 = Coord2(3.0 * self.padSize / 2.0 + self.padSpacing, self.padSize + 2.0 * self.padSpacing)
            centerMetalLineV2 = Coord2(5.0 * self.padSize / 2.0 + 2.0 * self.padSpacing, self.padSize + self.padSpacing)
            centerMetalLineI2 = Coord2(7.0 * self.padSize / 2.0 + 3.0 * self.padSpacing, self.padSize + self.padSpacing)

            if self.LineLayer == "MET_M1_6":
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                                   center=centerMetalLine1,
                                   box_size=(length1, self.w))
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                                   center=centerMetalLine2,
                                   box_size=(length2, self.w))
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                                   center=centerMetalLine3,
                                   box_size=(self.w, 2.0 * self.padSpacing+self.w))
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                                   center=centerMetalLineI1,
                                   box_size=(self.w, 4.0 * self.padSpacing + self.w))
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                                   center=centerMetalLineV1,
                                   box_size=(self.w, 4.0 * self.padSpacing + self.w))
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                                   center=centerMetalLineV2,
                                   box_size=(self.w, 2.0 * self.padSpacing + self.w))
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                                   center=centerMetalLineI2,
                                   box_size=(self.w, 2.0 * self.padSpacing + self.w))

            elif self.LineLayer == "MET_CH_1":
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                                   center=centerMetalLine1,
                                   box_size=(length1, self.w))
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                                   center=centerMetalLine2,
                                   box_size=(length2, self.w))
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                                   center=centerMetalLine3,
                                   box_size=(self.w, 2.0 * self.padSpacing+self.w))
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                                   center=centerMetalLineI1,
                                   box_size=(self.w, 4.0 * self.padSpacing + self.w))
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                                   center=centerMetalLineV1,
                                   box_size=(self.w, 4.0 * self.padSpacing + self.w))
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                                   center=centerMetalLineV2,
                                   box_size=(self.w, 2.0 * self.padSpacing + self.w))
                elems += Rectangle(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                                   center=centerMetalLineI2,
                                   box_size=(self.w, 2.0 * self.padSpacing + self.w))

            # Calculate inner length for 4 probes measurements:
            Innerlength = self.l - 8.0 * self.padSpacing - 3.0 * self.padSize

            # label
            font1 = TEXT_FONT_STANDARD.modified_copy()
            font1.spacing = 0.00
            font1.line_width = 0.1
            center = Coord2(800, self.padSize)
            label = PolygonText(layer=ProcessPurposeLayer(process=LABEL, purpose=COMMENT),
                                text = self.textLabel,
                                coordinate=center, font=font1,
                                height=self.padSize / 1.5)
            elems += label
        else:
            print("ERROR: !!!!! Metal line is to short")

        return elems


class psFTJs(Structure):
    
    FEcontactSize = PositiveNumberProperty(required=True)  # width of the metal via contact to junction
    MOcontactSize = PositiveNumberProperty(required=True)  # width of the metal via contact to junction
    viaDistance = PositiveNumberProperty(default=0.5)  # distance defining the wedge
    islandCcl = PositiveNumberProperty(default=0.5)  # distance between the mesa and the edge of the bottom electrode
    label_full = StringProperty(default="OFTJ")
    label = StringProperty(default="OFTJ")
    withoutFEMesa = BoolProperty(default=False)
    EBLCcl = PositiveNumberProperty(default=0.05)  # clearance (e-beam)

    def define_name(self):
        name = self.label
        return name
    
    def define_elements(self, elems):

        FEOpeningCenter = Coord2(0,0)
        MOViacenter = Coord2(102.5,0)    

        # Mesa definition, define device size
        # for leakage current measurements, don't pattern a Mesa.
        if self.withoutFEMesa == False:
            o1 = Coord2(-2*self.FEcontactSize/3, self.FEcontactSize)
            o2 = Coord2(2*self.FEcontactSize/3, self.FEcontactSize)
            o3 = Coord2(self.FEcontactSize , 2*self.FEcontactSize/3)
            o4 = Coord2(self.FEcontactSize , -2*self.FEcontactSize/3)
            o5 = Coord2(2*self.FEcontactSize/3 , -self.FEcontactSize)
            o6 = Coord2(-2*self.FEcontactSize/3 , -self.FEcontactSize)
            o7 = Coord2(-self.FEcontactSize , -2*self.FEcontactSize/3 )
            o8 = Coord2(-self.FEcontactSize , 2*self.FEcontactSize/3)
            
            if self.FEcontactSize <= fineLimit:

                elems += Boundary(layer=ProcessPurposeLayer(process=MET_TE_3_FINE, purpose=ETCH), 
                                  shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True))
            else:
                elems += Boundary(layer=ProcessPurposeLayer(process=MET_TE_3, purpose=ETCH), 
                                  shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True)) 

        # FE/MO island etch      
        p1 = Coord2(- self.FEcontactSize - self.islandCcl, - self.FEcontactSize - self.islandCcl)
        p2 = Coord2(self.viaDistance, - self.FEcontactSize - self.islandCcl)
        p3 = Coord2(2*self.viaDistance , (p2.y-17.5)/2.0) 
        p4 = Coord2(3*self.viaDistance , (p2.y-17.5)/2.0) 
        p5 = Coord2(55.0 , - 17.5)
        p6 = Coord2(150 , - 17.5)
        p7 = Coord2(150 , 17.5)  
        p8 = Coord2(55.0 , 17.5)
        p9 = Coord2(3*self.viaDistance , -(p2.y-17.5)/2.0) 
        p10 = Coord2(2*self.viaDistance , -(p2.y-17.5)/2.0) 
        p11 = Coord2(self.viaDistance, self.FEcontactSize + self.islandCcl)
        p12 = Coord2(- self.FEcontactSize - self.islandCcl, self.FEcontactSize + self.islandCcl)      
      
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                          shape=Shape(points=[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12], closed=True))        
        
        #Generate the left pad 
        q1 = Coord2(- 2*self.viaDistance , -(p2.y-17.5)/2.0)
        q2 = Coord2(- 3*self.viaDistance , -(p2.y-17.5)/2.0)
        q3 = Coord2(- 55.0 , 17.5)
        q4 = Coord2(- 150.0 , 17.5)
        q5 = Coord2(- 150.0 , - 17.5)
        q6 = Coord2(- 55.0 , - 17.5)
        q7 = Coord2(- 3*self.viaDistance , (p2.y-17.5)/2.0) 
        q8 = Coord2(- 2*self.viaDistance , (p2.y-17.5)/2.0)
        q9 = Coord2(- self.viaDistance, - self.FEcontactSize - self.islandCcl)
        q10 = Coord2(self.FEcontactSize + self.islandCcl, - self.FEcontactSize - self.islandCcl)        
        q11 = Coord2(self.FEcontactSize + self.islandCcl, self.FEcontactSize + self.islandCcl)
        q12 = Coord2(- self.viaDistance, self.FEcontactSize + self.islandCcl)
      
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                              shape=Shape(points=[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12], closed=True))    
        
        #Also create the ground pads
        r1 = Coord2(self.viaDistance, p2.y-4)
        r2 = Coord2(- self.viaDistance, p1.y-4)
        r3 = Coord2(- 2*self.viaDistance , q8.y-4)
        r4 = Coord2(- 3*self.viaDistance , q7.y-4)
        r5 = Coord2(- 55.0 , q6.y-4)
        r6 = Coord2(- 150.0 , q5.y-4)
        r7 = Coord2(- 150.0 , r6.y-48.5 )
        r8 = Coord2( 150.0 , r6.y-48.5)
        r9 = Coord2( 150.0 , p6.y-4)
        r10 = Coord2( 55 , - 21.5)
        r11 = Coord2(3*self.viaDistance , p4.y-4) 
        r12 = Coord2(2*self.viaDistance , p3.y-4) 
        
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                              shape=Shape(points=[r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12], closed=True))            

        s1 = Coord2(self.viaDistance, -p2.y+4)
        s2 = Coord2(- self.viaDistance, -p1.y+4)
        s3 = Coord2(- 2*self.viaDistance , -q8.y+4)
        s4 = Coord2(- 3*self.viaDistance , -q7.y+4)
        s5 = Coord2(- 55.0 , -q6.y+4)
        s6 = Coord2(- 150.0 , -q5.y+4)
        s7 = Coord2(- 150.0 , -r6.y+48.5 )
        s8 = Coord2( 150.0 , -r6.y+48.5)
        s9 = Coord2( 150.0 , -p6.y+4)
        s10 = Coord2( 55 , 21.5)
        s11 = Coord2(3*self.viaDistance , -p4.y+4) 
        s12 = Coord2(2*self.viaDistance , -p3.y+4) 
        
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                              shape=Shape(points=[s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12], closed=True)) 
        
        # Structure Fill MESA + BE
        
        f1=Coord2(-150, r6.y-68)
        f2=Coord2(-150, -r6.y+48.5)
        f3=Coord2(150, -r6.y+48.5)
        f4=Coord2(150, -r6.y)
        f5=Coord2(55, 21.5)
        f6=Coord2(3*self.viaDistance, -p4.y+4)
        f7=Coord2(2*self.viaDistance, -p3.y+4)
        f8=Coord2(self.viaDistance, -p2.y+4)
        #f9=Coord2(-self.viaDistance, -p1.y+4)
        f10=Coord2(p12.x-4,p12.y+4)
        f11=Coord2(p1.x-4,p1.y-4)
        #f10=Coord2(- 2*self.viaDistance , -q8.y+4)
        #f11=Coord2(- 2*self.viaDistance , q8.y-4)
        #f12=Coord2(-self.viaDistance, p1.y-4)
        f13=Coord2(self.viaDistance, p2.y-4)
        f14=Coord2(2*self.viaDistance, p3.y-4)
        f15=Coord2(3*self.viaDistance, p4.y-4)
        f16=Coord2(55, -21.5)
        f17=Coord2(150, r6.y)
        f18=Coord2(150, r6.y-68)
        
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_TE_3, purpose=ETCH),
                              shape=Shape(points=[f1, f2, f3, f4, f5, f6, f7, f8, f10, f11, f13, f14, f15, f16, f17, f18], closed=True))        
        
        w1=Coord2(f1.x - Ccl , f1.y)
        w2=Coord2(f2.x - Ccl , f2.y + Ccl)
        w3=Coord2(f3.x + Ccl , f3.y + Ccl)
        w4=Coord2(f4.x + Ccl , f4.y - Ccl)
        w5=Coord2(f5.x , f5.y - Ccl)
        w6=Coord2(f6.x , f6.y - Ccl)
        w7=Coord2(f7.x , f7.y - Ccl)
        w8=Coord2(f8.x , f8.y - Ccl)
        #w9=Coord2(f9.x , f9.y - Ccl)
        w10=Coord2(f10.x + Ccl , f10.y - Ccl)
        w11=Coord2(f11.x + Ccl , f11.y + Ccl)
        #w12=Coord2(f12.x , f12.y + Ccl)
        w13=Coord2(f13.x , f13.y + Ccl)
        w14=Coord2(f14.x , f14.y + Ccl)
        w15=Coord2(f15.x , f15.y + Ccl)
        w16=Coord2(f16.x , f16.y + Ccl)
        w17=Coord2(f17.x + Ccl , f17.y + Ccl)
        w18=Coord2(f18.x + Ccl , f18.y)
        
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_CH_1, purpose=ETCH),
                              shape=Shape(points=[w1, w2, w3, w4, w5, w6, w7, w8, w10, w11, w13, w14, w15, w16, w17, w18], closed=True))        

        j1=Coord2(50, 15)
        j2=Coord2(50,-15)
        j3=Coord2(self.FEcontactSize + 15, -2*self.FEcontactSize/3)
        j4=Coord2(self.FEcontactSize + 15, 2*self.FEcontactSize/3)
        
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_TE_3, purpose=ETCH),
                              shape=Shape(points=[j1, j2, j3, j4], closed=True))
        
              

        # Open passivation to bottom electrode
        elems += Rectangle(layer=ProcessPurposeLayer(process=VIA_CL_4, purpose=ETCH),
                               center=MOViacenter, box_size=(86.0, 26.0))
        elems += Rectangle(layer=ProcessPurposeLayer(process=VIA_SDG_5, purpose=ETCH),
                               center=MOViacenter, box_size=(90.0, 30.0))        
        # Metallise contact to bottom electrode:
        elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                               center=MOViacenter, box_size=(95.0, 35.0))
        # create q box for Boolean operation:
        '''elems += Rectangle(layer=ProcessPurposeLayer(process=BOOL, purpose=ETCH),
                               center=FEOpeningCenter, box_size=(305.0, 145.0))'''        
         
        # Open passivation to FE mesa
        if self.withoutFEMesa == False:
                if self.FEcontactSize <= fineLimit:

                    o1 = Coord2(-2*(self.FEcontactSize-self.EBLCcl)/3, (self.FEcontactSize-self.EBLCcl))
                    o2 = Coord2(2*(self.FEcontactSize-self.EBLCcl)/3, (self.FEcontactSize-self.EBLCcl))
                    o3 = Coord2((self.FEcontactSize-self.EBLCcl) , 2*(self.FEcontactSize-self.EBLCcl)/3)
                    o4 = Coord2((self.FEcontactSize-self.EBLCcl) , -2*(self.FEcontactSize-self.EBLCcl)/3)
                    o5 = Coord2(2*(self.FEcontactSize-self.EBLCcl)/3 , -(self.FEcontactSize-self.EBLCcl))
                    o6 = Coord2(-2*(self.FEcontactSize-self.EBLCcl)/3 , -(self.FEcontactSize-self.EBLCcl))
                    o7 = Coord2(-(self.FEcontactSize-self.EBLCcl) , -2*(self.FEcontactSize-self.EBLCcl)/3 )
                    o8 = Coord2(-(self.FEcontactSize-self.EBLCcl) , 2*(self.FEcontactSize-self.EBLCcl)/3)                

                    elems += Boundary(layer=ProcessPurposeLayer(process=VIA_SDG_5_FINE, purpose=ETCH), 
                                  shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True))
                else:

                    o1 = Coord2(-2*(self.FEcontactSize-Ccl)/3, (self.FEcontactSize-Ccl))
                    o2 = Coord2(2*(self.FEcontactSize-Ccl)/3, (self.FEcontactSize-Ccl))
                    o3 = Coord2((self.FEcontactSize-Ccl) , 2*(self.FEcontactSize-Ccl)/3)
                    o4 = Coord2((self.FEcontactSize-Ccl) , -2*(self.FEcontactSize-Ccl)/3)
                    o5 = Coord2(2*(self.FEcontactSize-Ccl)/3 , -(self.FEcontactSize-Ccl))
                    o6 = Coord2(-2*(self.FEcontactSize-Ccl)/3 , -(self.FEcontactSize-Ccl))
                    o7 = Coord2(-(self.FEcontactSize-Ccl) , -2*(self.FEcontactSize-Ccl)/3 )
                    o8 = Coord2(-(self.FEcontactSize-Ccl) , 2*(self.FEcontactSize-Ccl)/3)                


                    elems += Boundary(layer=ProcessPurposeLayer(process=VIA_SDG_5, purpose=ETCH), 
                                  shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True))
        # for leakage current measurements, open above the Mesa with layer 4
        if self.withoutFEMesa == True:
                if self.FEcontactSize <= fineLimit:

                    o1 = Coord2(-2*(self.FEcontactSize-self.EBLCcl)/3, (self.FEcontactSize-self.EBLCcl))
                    o2 = Coord2(2*(self.FEcontactSize-self.EBLCcl)/3, (self.FEcontactSize-self.EBLCcl))
                    o3 = Coord2((self.FEcontactSize-self.EBLCcl) , 2*(self.FEcontactSize-self.EBLCcl)/3)
                    o4 = Coord2((self.FEcontactSize-self.EBLCcl) , -2*(self.FEcontactSize-self.EBLCcl)/3)
                    o5 = Coord2(2*(self.FEcontactSize-self.EBLCcl)/3 , -(self.FEcontactSize-self.EBLCcl))
                    o6 = Coord2(-2*(self.FEcontactSize-self.EBLCcl)/3 , -(self.FEcontactSize-self.EBLCcl))
                    o7 = Coord2(-(self.FEcontactSize-self.EBLCcl) , -2*(self.FEcontactSize-self.EBLCcl)/3 )
                    o8 = Coord2(-(self.FEcontactSize-self.EBLCcl) , 2*(self.FEcontactSize-self.EBLCcl)/3)                

                    elems += Boundary(layer=ProcessPurposeLayer(process=VIA_CL_4_FINE, purpose=ETCH),       
                                      shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True))
                else:

                    o1 = Coord2(-2*(self.FEcontactSize-Ccl)/3, (self.FEcontactSize-Ccl))
                    o2 = Coord2(2*(self.FEcontactSize-Ccl)/3, (self.FEcontactSize-Ccl))
                    o3 = Coord2((self.FEcontactSize-Ccl) , 2*(self.FEcontactSize-Ccl)/3)
                    o4 = Coord2((self.FEcontactSize-Ccl) , -2*(self.FEcontactSize-Ccl)/3)
                    o5 = Coord2(2*(self.FEcontactSize-Ccl)/3 , -(self.FEcontactSize-Ccl))
                    o6 = Coord2(-2*(self.FEcontactSize-Ccl)/3 , -(self.FEcontactSize-Ccl))
                    o7 = Coord2(-(self.FEcontactSize-Ccl) , -2*(self.FEcontactSize-Ccl)/3 )
                    o8 = Coord2(-(self.FEcontactSize-Ccl) , 2*(self.FEcontactSize-Ccl)/3)                


                    elems += Boundary(layer=ProcessPurposeLayer(process=VIA_CL_4, purpose=ETCH), 
                                      shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True)) 

        # create the full label
        full_labelHeight = 10
        font1 = TEXT_FONT_STANDARD.modified_copy()
        font1.spacing = 0.00
        font1.line_width = 0.1
        label_full = PolygonText(layer=ProcessPurposeLayer(process=FULL_LABEL, purpose=COMMENT), 
                                text=self.label_full,
                                coordinate=Coord2(0, -75.0), 
                                font=font1, 
                                height=full_labelHeight)
        elems += label_full        
        # Also add the short label
        labelHeight = 50
        font1 = TEXT_FONT_STANDARD.modified_copy()
        font1.spacing = 0.00
        font1.line_width = 0.1
        label = PolygonText(layer=ProcessPurposeLayer(process=LABEL, purpose=COMMENT), 
                            text=self.label,
                            coordinate=Coord2(0, -25.0),
                            font=font1, 
                            height=labelHeight)
        elems += label
        
        
        elems += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                               center=Coord2(0, -78.75), box_size=(300, 17.5))        

        return elems
    
class PicoProbe(Structure):
    FEcontactSize = PositiveNumberProperty(required=True)  # width(radius) of the metal via contact to junction
    label = StringProperty(default="OFTJ")
    label_full = StringProperty(default="OFTJ")
    withoutFEMesa = BoolProperty(default=False)

    def define_name(self):
        name = self.label
        return name
    
    def define_elements(self, elems):

        FEOpeningCenter = Coord2(0,0)
        MOViacenter = Coord2(102.5,0)    

        # Top electrode (Mesa) definition, define device size
        # for leakage current measurements, don't pattern a Mesa.
        if self.withoutFEMesa == False:
            o1 = Coord2(-2*self.FEcontactSize/3, self.FEcontactSize)
            o2 = Coord2(2*self.FEcontactSize/3, self.FEcontactSize)
            o3 = Coord2(self.FEcontactSize , 2*self.FEcontactSize/3)
            o4 = Coord2(self.FEcontactSize , -2*self.FEcontactSize/3)
            o5 = Coord2(2*self.FEcontactSize/3 , -self.FEcontactSize)
            o6 = Coord2(-2*self.FEcontactSize/3 , -self.FEcontactSize)
            o7 = Coord2(-self.FEcontactSize , -2*self.FEcontactSize/3 )
            o8 = Coord2(-self.FEcontactSize , 2*self.FEcontactSize/3)             
            
            elems += Boundary(layer=ProcessPurposeLayer(process=MET_TE_3, purpose=ETCH), 
                                 shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True)) 
             
       
        # Pad (Metal Line Top Electrode) for top electrode (mesa) / or just a direct top electrode
        o1 = Coord2(-2*(self.FEcontactSize+Ccl_big)/3, (self.FEcontactSize+Ccl_big))
        o2 = Coord2(2*(self.FEcontactSize+Ccl_big)/3, (self.FEcontactSize+Ccl_big))
        o3 = Coord2((self.FEcontactSize+Ccl_big) , 2*(self.FEcontactSize+Ccl_big)/3)
        o4 = Coord2((self.FEcontactSize+Ccl_big) , -2*(self.FEcontactSize+Ccl_big)/3)
        o5 = Coord2(2*(self.FEcontactSize+Ccl_big)/3 , -(self.FEcontactSize+Ccl_big))
        o6 = Coord2(-2*(self.FEcontactSize+Ccl_big)/3 , -(self.FEcontactSize+Ccl_big))
        o7 = Coord2(-(self.FEcontactSize+Ccl_big) , -2*(self.FEcontactSize+Ccl_big)/3 )
        o8 = Coord2(-(self.FEcontactSize+Ccl_big) , 2*(self.FEcontactSize+Ccl_big)/3)                
    
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH), 
                                      shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True))       
        
        '''        
        # HZO layer (Boolean Operator)
        o1 = Coord2(-2*(self.FEcontactSize+Ccl)/3, (self.FEcontactSize+Ccl))
        o2 = Coord2(2*(self.FEcontactSize+Ccl)/3, (self.FEcontactSize+Ccl))
        o3 = Coord2((self.FEcontactSize+Ccl) , 2*(self.FEcontactSize+Ccl)/3)
        o4 = Coord2((self.FEcontactSize+Ccl) , -2*(self.FEcontactSize+Ccl)/3)
        o5 = Coord2(2*(self.FEcontactSize+Ccl)/3 , -(self.FEcontactSize+Ccl))
        o6 = Coord2(-2*(self.FEcontactSize+Ccl)/3 , -(self.FEcontactSize+Ccl))
        o7 = Coord2(-(self.FEcontactSize+Ccl) , -2*(self.FEcontactSize+Ccl)/3 )
        o8 = Coord2(-(self.FEcontactSize+Ccl) , 2*(self.FEcontactSize+Ccl)/3)                
        
        elems += Boundary(layer=ProcessPurposeLayer(process=BOOL, purpose=ETCH), 
                        shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True))        
        '''
         
        # HZO Via between Metal line (M1) and top electrode / Open passivation to FE mesa
        if self.withoutFEMesa == False:

            o1 = Coord2(-2*(self.FEcontactSize-Ccl_big)/3, (self.FEcontactSize-Ccl_big))
            o2 = Coord2(2*(self.FEcontactSize-Ccl_big)/3, (self.FEcontactSize-Ccl_big))
            o3 = Coord2((self.FEcontactSize-Ccl_big) , 2*(self.FEcontactSize-Ccl_big)/3)
            o4 = Coord2((self.FEcontactSize-Ccl_big) , -2*(self.FEcontactSize-Ccl_big)/3)
            o5 = Coord2(2*(self.FEcontactSize-Ccl_big)/3 , -(self.FEcontactSize-Ccl_big))
            o6 = Coord2(-2*(self.FEcontactSize-Ccl_big)/3 , -(self.FEcontactSize-Ccl_big))
            o7 = Coord2(-(self.FEcontactSize-Ccl_big) , -2*(self.FEcontactSize-Ccl_big)/3 )
            o8 = Coord2(-(self.FEcontactSize-Ccl_big) , 2*(self.FEcontactSize-Ccl_big)/3)                
                            
            
            elems += Boundary(layer=ProcessPurposeLayer(process=VIA_SDG_5, purpose=ETCH), 
                            shape=Shape(points=[o1, o2, o3, o4, o5, o6, o7, o8], closed=True))
                
        # for leakage current measurements, open above the Mesa
       
        labelHeight = 8
        # Generate pad for big FEsizes
        # Create the L-shape pad
        s1 = Coord2(85, -75)
        s2 = Coord2(170, -75)
        s3 = Coord2(170, 125)
        s4 = Coord2(160, 125)  
        s5 = Coord2(160, 78)
        s6 = Coord2(85, 78)             
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                                  shape=Shape(points=[s1, s2, s3, s4, s5, s6], closed=True))
        # Create the GND pad
        t1 = Coord2(- 80, 83)
        t2 = Coord2(155, 83)
        t3 = Coord2(155, 120)
        t4 = Coord2(- 80, 120)            
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                                  shape=Shape(points=[t1, t2, t3, t4], closed=True))	
        
        # Metal contact to channel(VO2)
        b1 = Coord2(95, -65)
        b2 = Coord2(160, -65)
        b3 = Coord2(160, 68)
        b4 = Coord2(95, 68)            
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_SD_2, purpose=ETCH),
                                  shape=Shape(points=[b1, b2, b3, b4], closed=True))	        
        
        # Passivation opening VlA from Ml to HZO VIA for channel contact
        p1 = Coord2(100, -60)
        p2 = Coord2(155, -60)
        p3 = Coord2(155, 63)
        p4 = Coord2(100, 63)            
        elems += Boundary(layer=ProcessPurposeLayer(process=VIA_SDG_5, purpose=ETCH),
                                  shape=Shape(points=[p1, p2, p3, p4], closed=True))          
        
        # HZO opening VlA to channel contact     
        h1 = Coord2(105, -55)
        h2 = Coord2(105, 58)
        h3 = Coord2(150, 58)
        h4 = Coord2(150, -55)           
        elems += Boundary(layer=ProcessPurposeLayer(process=VIA_CL_4, purpose=ETCH),
                                  shape=Shape(points=[h1, h2, h3, h4], closed=True))        
        
        #Filling
        
        f1 = Coord2(s6.x-Ccl_big, s6.y)
        f2 = Coord2(2*(self.FEcontactSize+Ccl_big)/3+Ccl_big, (self.FEcontactSize+Ccl_big)+Ccl_big)
        f3 = Coord2(2*(self.FEcontactSize+Ccl_big)/3+Ccl_big, (self.FEcontactSize+Ccl_big)+Ccl_big)
        f4 = Coord2((self.FEcontactSize+Ccl_big)+Ccl_big , 2*(self.FEcontactSize+Ccl_big)/3+Ccl_big)
        f5 = Coord2((self.FEcontactSize+Ccl_big)+Ccl_big , -2*(self.FEcontactSize+Ccl_big)/3-Ccl_big)
        f6 = Coord2(2*(self.FEcontactSize+Ccl_big)/3+Ccl_big , -(self.FEcontactSize+Ccl_big)-Ccl_big)
        f7 = Coord2(-2*(self.FEcontactSize+Ccl_big)/3-Ccl_big , -(self.FEcontactSize+Ccl_big)-Ccl_big)
        f8 = Coord2(-(self.FEcontactSize+Ccl_big)-Ccl_big , -2*(self.FEcontactSize+Ccl_big)/3-Ccl_big)
        f9 = Coord2(-(self.FEcontactSize+Ccl_big)-Ccl_big , 2*(self.FEcontactSize+Ccl_big)/3+Ccl_big)
        f10 = Coord2(-2*(self.FEcontactSize+Ccl_big)/3-Ccl_big, (self.FEcontactSize+Ccl_big)+Ccl_big)        
        f11 = Coord2(t4.x, s6.y)
        f12 = Coord2(t4.x, t1.y-158)
        f13 = Coord2(s1.x-Ccl_big, t1.y-158)
        
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH), 
                                      shape=Shape(points=[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13], closed=True))        
       
        elems += Boundary(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH), 
                                      shape=Shape(points=[f10, f11, f1, f2], closed=True))
       
        # create the full label
        full_labelHeight = 25
        font1 = TEXT_FONT_STANDARD.modified_copy()
        font1.spacing = 0.00
        font1.line_width = 0.1
        label_full = PolygonText(layer=ProcessPurposeLayer(process=FULL_LABEL, purpose=COMMENT), 
                                text=self.label_full,
                                coordinate=Coord2(40, 115.0), 
                                font=font1, 
                                height=full_labelHeight)
        elems += label_full        
        # Also add the short label
        labelHeight = 70
        font1 = TEXT_FONT_STANDARD.modified_copy()
        font1.spacing = 0.00
        font1.line_width = 0.1
        label = PolygonText(layer=ProcessPurposeLayer(process=LABEL, purpose=COMMENT), 
                            text=self.label,
                            coordinate=Coord2(90.0, 7.0),
                            transformation=Rotation(rotation_center=(90.0, 7.0), rotation=90.0),
                            font=font1, 
                            height=labelHeight)
        elems += label

        return elems

# Create boxes
PETs = Structure(name="PETs")
ListOfPETs = []
my_layout = Structure(name="myLayout")

deviceNbr = 0

# Create a list to store the output data with required columns
output_data = []

# Die map
# f = open('DieMap.txt', 'w')
# writeLayoutStyleFile(process_layer_map=process_layer_map,purpose_datatype_map=purpose_datatype_map)

# ---------------------------------------------psFTJs starts here-----------------------------------------------

# add structure
my_PETs_ps = Structure(name="psFTJs")

# ---------------------------------- Parameter Definitions Reference Structures
# FE_contact are all contact sizes for the metal directly on the ferroelectric material (FE) where the junction is
#FE_Contact_Sizes = [30.0,20.0,10.0,5.0,2.0,1.0,0.8,0.6,0.5,0.4,0.3,0.2,0.15]
FE_Contact_Sizes = [30.0,28.0,26.0,25.0,24.0,22.0,21.0,20.0,19.0,18.0,16.0,15.0,14.0,12.0,10.0,8.0,7.0,6.0]
numberOfDevices = 5


# Position
dX = 304
dY = 161.5
curX = 0.0
curY = 0.0
nX = 0
nY = 0

#Adjust EBLCcl values
#EBLCcl_values = [0.02,0.05,0.1,0.2]

#islandCcl_values = [0.2,0.4,0.6]
islandCcl_values = [3.0]



for i in range(numberOfDevices):
    curX = 0
    for FESize in FE_Contact_Sizes:
        #Determine the appropriate EBLCcl value based on FE_Contact_Sizes
        EBLCcl_value=0.2
        if FESize <= 0.2:
            EBLCcl_value = 0.05
        elif FESize == 0.3:
            EBLCcl_value = 0.1              
        elif FESize == 0.4:
            EBLCcl_value = 0.1      
        elif FESize == 0.5:
            EBLCcl_value = 0.1         

        #else:
            #EBLCcl_value = EBLCcl_values[nX % len(EBLCcl_values)]
        
        for island_value in islandCcl_values: 
            full_textLabel = 'ID%i %.0f EBcl=%.2f iscl=%.2f' % (deviceNbr, FESize * 1000, EBLCcl_value, island_value) 
            textLabel = '%i' % (deviceNbr)    
            temp1 = psFTJs(FEcontactSize = FESize / 2.0,  # radius
                           MOcontactSize = 5.0,
                           viaDistance = 10.0,
                           islandCcl = island_value,
                           EBLCcl = EBLCcl_value,  # clearance (e-beam)
                           label = textLabel,
                           label_full = full_textLabel)
            my_PETs_ps += SRef(temp1, position=(curX, curY))
            curX += dX
            deviceNbr += 1
    curY += dY
    nY += 1       

# ---------------------------------------------psFTJs ENDs here----------------------------------------------------

# ---------------------------------------------PicoProbe starts here-----------------------------------------------

# add structure
my_PETs_O = Structure(name="PicoProbe")

# ---------------------------------- Parameter Definitions Reference Structures
# FE_contact are all contact sizes for the metal directly on the ferroelectric material (FE) where the junction is
FE_Contact_Sizes = [120.0,100.0,80.0,60.0,50.0,40.0,20.0]
#FE_Contact_Sizes = [120.0,80.0,40.0]
#numberOfDevices = 7
numberOfDevices = 3

# Position
dX = 255.0
dY = 200.0
curX = 0.0
curY = 0.0
nX = 0
nY = 0

islandCcl_values = [0.2]


for i in range(numberOfDevices):
    curX = 0
    for FESize in FE_Contact_Sizes:
        #Determine the appropriate Ccl value based on FE_Contact_Sizes
        #EBLCcl_value=0.2
       
        for n in range(3):
            full_textLabel = 'ID{0} D{1}'.format(deviceNbr, FESize)
            textLabel = '%i' % (deviceNbr)    
            output_data.append([deviceNbr, "PicoProbe", 'D{0}'.format(FESize)])         
            temp1 = PicoProbe(FEcontactSize = FESize / 2.0,  # radius
                           label = textLabel,
                           label_full = full_textLabel)
            my_PETs_O += SRef(temp1, position=(curX, curY))
            curX += dX
            deviceNbr += 1
    curY += dY
    nY += 1 

# ---------------------------------------------PicoProbe ENDs here----------------------------------------------------



# ---------------------------------------------FeCAP starts here-----------------------------------------------

# add structure
my_PETs_50CAP = Structure(name="FeCAP_50")
my_PETs_10CAP = Structure(name="FeCAP_10")

# ---------------------------------- Parameter Definitions Reference Structures
# FE_contact are all contact sizes for the metal directly on the ferroelectric material (FE) where the junction is
FE_Contact_Sizes = [50.0] * 32
numberOfDevices = 22

# Position
dX = 304
#dY = 121.5
dY = 171.5
curX = 0.0
curY = 0.0
nX = 0
nY = 0

#Adjust Ccl values
#EBLCcl_values = [0.02,0.05,0.1,0.2]

#islandCcl_values = [0.2,0.4,0.6]
islandCcl_values = [2.0]


for i in range(numberOfDevices):
    curX = 0
    for FESize in FE_Contact_Sizes:
        #Determine the appropriate Ccl value based on FE_Contact_Sizes
        '''
        EBLCcl_value=0.2
        if FESize <= 0.2:
            EBLCcl_value = 0.05
        elif FESize == 0.3:
            EBLCcl_value = 0.1              
        elif FESize == 0.4:
            EBLCcl_value = 0.1      
        elif FESize == 0.5:
            EBLCcl_value = 0.1         

        #else:
            #EBLCcl_value = EBLCcl_values[nX % len(EBLCcl_values)]
        '''
        for island_value in islandCcl_values:
            full_textLabel = 'ID%i %.0f' % (deviceNbr, FESize)
            textLabel = 'ID%i' % (deviceNbr)
            output_data.append([deviceNbr, "FeCAP_50", '%.0f' % (FESize * 1000)]) 
            temp1 = FeCAP(FEcontactSize = FESize / 2.0,  # radius
                           MOcontactSize = 5.0,
                           viaDistance = 10.0,
                           islandCcl = island_value,
                           #Ccl = EBLCcl_value,  # clearance (e-beam)
                           label = textLabel,
                           label_full = full_textLabel)
            my_PETs_50CAP += SRef(temp1, position=(curX, curY))
            curX += dX
            deviceNbr += 1
    curY += dY
    nY += 1       


FE_Contact_Sizes = [10.0] * 32
numberOfDevices = 22

curX = 0.0
curY = 0.0
nX = 0
nY = 0

for i in range(numberOfDevices):
    curX = 0
    for FESize in FE_Contact_Sizes:
        #Determine the appropriate Ccl value based on FE_Contact_Sizes
        '''
        EBLCcl_value=0.2
        if FESize <= 0.2:
            EBLCcl_value = 0.05
        elif FESize == 0.3:
            EBLCcl_value = 0.1              
        elif FESize == 0.4:
            EBLCcl_value = 0.1      
        elif FESize == 0.5:
            EBLCcl_value = 0.1         

        #else:
            #EBLCcl_value = EBLCcl_values[nX % len(EBLCcl_values)]
        '''
        for island_value in islandCcl_values:
            full_textLabel = 'ID%i %.0f' % (deviceNbr, FESize)
            textLabel = 'ID%i' % (deviceNbr)
            output_data.append([deviceNbr, "FeCAP_10", '%.0f' % (FESize * 1000)]) 
            temp1 = FeCAP(FEcontactSize = FESize / 2.0,  # radius
                           MOcontactSize = 5.0,
                           viaDistance = 10.0,
                           islandCcl = island_value,
                           #Ccl = EBLCcl_value,  # clearance (e-beam)
                           label = textLabel,
                           label_full = full_textLabel)
            if deviceNbr not in [1032,1033,1000,1001,968,969]:
                my_PETs_10CAP += SRef(temp1, position=(curX, curY))           
            curX += dX
            deviceNbr += 1
    curY += dY
    nY += 1           

# ---------------------------------------------FeCAP ENDs here----------------------------------------------------


# --------------------------------------MetalLine structures starts here-----------------------------------------
# add structure
my_PETs_ML = Structure(name="Metallines")
ML = [(4000, 5), (3000, 5), (2000, 5), (4000, 1.35), (3000, 1.35), (2000, 1.35)]
padSize = 80.0
for si in ML:
    label="ID%i MLM1 %i x %i" % (deviceNbr, si[0], si[1])
    my_PETs_Metallines = Structure(name=label)
    metalLine = FourProbesMetalLines(l=si[0], w=si[1],
                                     padSize=padSize, 
                                     LineLayer="MET_M1_6", 
                                     textLabel = label)
    my_PETs_Metallines += SRef(metalLine, position=(0,0))
    ListOfPETs.append(my_PETs_Metallines)
    deviceNbr += 1

ML = [(4000, 32), (3000, 32), (2000, 32), (4000, 1.35), (3000, 1.35), (2000, 1.35)]
for si in ML:
    label="ID%i MLBE %i x %i" % (deviceNbr, si[0], si[1])
    my_PETs_Metallines = Structure(name=label)
    metalLine = FourProbesMetalLines(l=si[0], w=si[1],
                                     padSize=padSize, 
                                     LineLayer="MET_CH_1", 
                                     textLabel = label)
    my_PETs_Metallines += SRef(metalLine, position=(0,0))
    ListOfPETs.append(my_PETs_Metallines)
    deviceNbr += 1

my_PETs_ML +=SRef(ListOfPETs[0], position=(-6000,5400))
my_PETs_ML +=SRef(ListOfPETs[1], position=(-6000,5650))
my_PETs_ML +=SRef(ListOfPETs[2], position=(-6000,5900))
my_PETs_ML +=SRef(ListOfPETs[3], position=(-6000,6150))
my_PETs_ML +=SRef(ListOfPETs[4], position=(-6000,6400))
my_PETs_ML +=SRef(ListOfPETs[5], position=(-6000,6650))


my_PETs_ML +=SRef(ListOfPETs[8], position=(-3900,5400))
my_PETs_ML +=SRef(ListOfPETs[7], position=(-4300,5650))
my_PETs_ML +=SRef(ListOfPETs[6], position=(-4800,5900))
my_PETs_ML +=SRef(ListOfPETs[11], position=(-3900,6150))
my_PETs_ML +=SRef(ListOfPETs[10], position=(-4300,6400))
my_PETs_ML +=SRef(ListOfPETs[9], position=(-4800,6650))

# ---------------------------------------MetalLine structures ends here------------------------------------------
#------------------------------------------------Profilometer starts HERE----------------------------------------------
PROFILES = Structure(name="PROFILES")
boxsize = 130.0
dX = -30.0
processlayer = [MET_TE_3, MET_CH_1, VIA_SDG_5, VIA_CL_4, MET_M1_6]
curX=0

for process in processlayer:
    temp = Rectangle(layer=ProcessPurposeLayer(process=process,purpose=ETCH),center=Coord2(curX,0),
                     box_size=[boxsize,boxsize])
    PROFILES += temp
    temp = Rectangle(layer=ProcessPurposeLayer(process=process,purpose=ETCH),center=Coord2(curX+boxsize,0),
                     box_size=[boxsize,boxsize])
    PROFILES += temp
    temp = Rectangle(layer=ProcessPurposeLayer(process=process,purpose=ETCH),center=Coord2(curX+boxsize,-boxsize),
                     box_size=[boxsize,boxsize])
    PROFILES += temp
    
    blockText = process.extension
    font1 = TEXT_FONT_STANDARD.modified_copy()
    font1.spacing = 0.1
    font1.line_width = 0.1
    temp = PolygonText(layer=ProcessPurposeLayer(process=LABEL, purpose=COMMENT), text="%s" % (blockText),
                       coordinate=Coord2(curX+boxsize/2,boxsize),alignment=(TEXT_ALIGN_CENTER, TEXT_ALIGN_BOTTOM),
                        font=font1, height=20.0)
    PROFILES += temp
    curX += dX + 2*boxsize

#------------------------------------------------Profilometer ends HERE----------------------------------------------

GroundPads = Structure(name="GroundPads")

GroundPads += Rectangle(layer=ProcessPurposeLayer(process=MET_M1_6, purpose=ETCH),
                               center=Coord2(119,20),
                               box_size=(238,40));


#------------------------------------------------Main Label starts HERE----------------------------------------------
labelMain = Structure(name="LabelMain")

font1 = TEXT_FONT_STANDARD.modified_copy()
font1.spacing = 0.1
font1.line_width = 0.1
temp = PolygonText(layer=ProcessPurposeLayer(process=LABEL, purpose=COMMENT), text="Paul",
                   coordinate=Coord2(0,80), alignment=(TEXT_ALIGN_CENTER, TEXT_ALIGN_BOTTOM),
                   font=font1, height=140.0)
labelMain += temp
#temp = PolygonText(layer=ProcessPurposeLayer(process=LABEL, purpose=COMMENT), text="SecondLineText",
                   #coordinate=Coord2(0,-30), alignment=(TEXT_ALIGN_CENTER, TEXT_ALIGN_BOTTOM),
                   #font=font1, height=90.0)
#labelMain += temp
#------------------------------------------------Main Label ends HERE----------------------------------------------

# Import die bounding box (BRNC template)
DWL_Marker = placeGDSonWafer(GDSfilePath="laserwriter_markers/DWL_AlignmentMarks.gds", process_layer_map={DWLMARKS: 1},
                             purpose_datatype_map={VIEW: 0})
my_layout += ARef(DWL_Marker, origin=(-5000.0, -5000), period=(5000.0, 5000.0), n_o_periods=(3, 3))

# Define the positions for optical markers
OPT_Marker = placeGDSonWafer(GDSfilePath="laserwriter_markers/optical_markers.gds", process_layer_map={OPTMARKS_L1: 1, OPTMARKS_L2: 2},
                             purpose_datatype_map={VIEW: 0})
optical_marker_positions = [ 
    (-500, 5200), (-2500, 5200), (-1500, 5200), (-500, -5200), (-2500, -5200), (-1500, -5200), (-5200, 500), (-5200, 2500), (-5200, 1500), 
    (-5200, -500), (-5200, -2500), (-5200, -1500), (500, 5200), (2500, 5200), (1500, 5200), (500, -5200), (2500, -5200), (1500, -5200), (5200, 500), (5200, 2500), (5200, 1500), (5200, -500), (5200, -2500), (5200, -1500),
    (-5200, -3500), (-5200, -4500), (-5200, 3500), (-5200, 4500), (5200, -3500), (5200, -4500), (5200, 3500), (5200, 4500), (-3500, -5200), (-4500, -5200), (4500, -5200), (3500, -5200),(-3500, 5200), (-4500, 5200), (4500, 5200), (3500, 5200)
]  # markers if needed:(-150, 400), (-150, -400),
for pos in optical_marker_positions:
    my_layout += SRef(OPT_Marker, position=pos)

#Position  FeCAP devices in top cell
my_layout += SRef(my_PETs_10CAP, position = Coord2(-4700, -635.00))
my_layout += SRef(my_PETs_50CAP, position = Coord2(-4700, -4535.00))

#Position PicoProbe
my_layout += SRef(my_PETs_O, position=Coord2(-750.0, 4200))

#Position  ps FTJ devices in top cell
my_layout += SRef(my_PETs_ps, position = Coord2(-650.0, 3300))

#Position metal lines
my_layout += SRef(my_PETs_ML, position=Coord2(1200,-2080))

#Position Labels
my_layout += SRef(labelMain, position=(-1125,3250)) # , transformation=Rotation(rotation=90.0)
#Position Profiles
my_layout += SRef(PROFILES, position=(-1250,3600), transformation=Rotation(rotation=90.0))

#my_layout += ARef(reference=GroundPads, origin=Coord2(-9565,-4120), period=Coord2(252, 0), n_o_periods=(36, 0))


##----------------------------------------------------------------------------------
# EXPORT LAYOUT
##----------------------------------------------------------------------------------
# my_lib += my_layout
# my_output = FileOutputGdsii(FILENAME)
# my_output.write(my_lib)
my_layout.write_gdsii("%s" % (FILENAME), unit=1E-6, grid=MYGRIDSIZE)

# Create output file path
#output_dir = "AssignedDeviceID"
#output_file = "Assigned Device ID.csv" # f"{output_dir}/output.csv"

# Write to CSV file
#with open(output_file, 'wb') as csvfile:
#    csv_writer = csv.writer(csvfile)  # Create a CSV writer object
#    csv_writer.writerow(['ID', 'Structure name', 'Variables'])  # Write the header
#    csv_writer.writerows(output_data)  # Write the data rows
