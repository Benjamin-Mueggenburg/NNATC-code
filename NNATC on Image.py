from utils import label_map_util

from utils import visualization_utils as vis_util

#OWN FUNCTIONS
import NNATC_model_functions as model_func
from nnatc_classes import Car

from PIL import Image
import PIL.ImageDraw as ImageDraw

def DrawBox(image, xy, colour, thickness=5):
    draw = ImageDraw.Draw(image)
    xmin, ymin, xmax, ymax = xy
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    #draw.rectangle(xy, fill, outline)
    #(left, top, right, bottom) = xy
    #draw.line([(left, top), (left, bottom), (right, bottom),
             #(right, top), (left, top)], width=thickness, fill=colour)
    draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=colour)
    

def ReverseCords(xy):
    new_xy = (xy[0], xy[3], xy[2], xy[1])
    return new_xy

#Converts the cords of the person box - which are relative to the absolute cords in the image
def ConvertRelativeToAbsolute(box, car_box):
    xmin, ymin, xmax, ymax = box
    
    xmin_car, ymin_car, xmax_car, ymax_car = car_box

    print("Person box: {}".format(box))
    print("Car box: {}".format(car_box))
    
    
    
    new_xmin = xmin_car + xmin
    new_ymin = ymin_car + ymin #ymax_car + ymin  
    
    new_xmax = xmin_car + xmax  
    new_ymax = ymin_car + ymax  #ymax_car + ymax 
    print("New Box: {}".format((new_xmin, new_ymin, new_xmax, new_ymax)))
    return (new_xmin, new_ymin, new_xmax, new_ymax)
min_score_thresh_car = 0.8
max_boxes_to_draw_car = 30

min_score_thresh_person = 0.5
max_boxes_to_draw_person = 5

image_path = "test_images/negative/negativetestimage.JPG"

doCarDetection = True
doPersonDetection = True
#Slow detection works
slowDetection = False

#loads image to be saved in car objects
image = Image.open(image_path)
if doCarDetection:

    car_boxes, car_scores = model_func.DetectCarsInImage(image_path,useNormalised=False)

    cars = []

 
    #Loops through all the boxes and if it meets the criteria - of min_score_thresh
    for i in range(min(max_boxes_to_draw_car, car_boxes.shape[0])):
        if car_scores is None or car_scores[i] > min_score_thresh_car:
          box = tuple(car_boxes[i].tolist())
          #print(box)
          #Car is a class found in nnatc_classes.py
          cars.append(Car(box, car_scores[i]))
    #add croped images
    for car in cars:
      car.AddCroppedImage(image)

###Run inference graph on each cropped image of the car
if doPersonDetection and not slowDetection:
    for car in cars:
            person_boxes, person_scores = model_func.DetectPeopleOnPhonesInImage(car.croppedimage, useNormalised=False, useImagePath=False)
            #For every result, check if it makes the threshold
            for i in range(min(max_boxes_to_draw_person, person_boxes.shape[0])):
                if person_scores is None or person_scores[i] > min_score_thresh_person:
                    box = tuple(person_boxes[i].tolist())
                    print("Estimated Location of person: {}".format(box))
                    #print(box)
                    #Add the cords of the box and the location of the person to the car object
                    car.people.append(box)
                    car.people_scores.append(person_scores[i])

#This does the exact as just above but on the whole image instead of the cropped version          
if doPersonDetection and slowDetection:
    person_boxes, person_scores = model_func.DetectPeopleOnPhonesInImage(image_path, useNormalised=False)
    parsed_boxes = []
    parsed_scores = []
    for i in range(min(max_boxes_to_draw_person, person_boxes.shape[0])):
            if person_scores is None or person_scores[i] > min_score_thresh_person:
                box = tuple(person_boxes[i].tolist())
                parsed_boxes.append(box)
                parsed_scores.append(person_scores[i])

if doCarDetection:           
    #Draw images on car
    for car in cars:
        #If no one is distracted 
        if car.people == []:
            #Draw box for car - green outline
            DrawBox(image, car.box, colour=(0,255,0), thickness=5)
        else:
            #Draw box for car - red outline
            DrawBox(image, car.box, colour=(255,0,0), thickness=15)

#Draw boxes for people - fast version
if doPersonDetection and not slowDetection:
    for car in cars:
        for person_box in car.people:
            
            #Draw box for person - Blue outline
            DrawBox(image, ReverseCords(ConvertRelativeToAbsolute(person_box, car.box)), colour=(0,0,255), thickness=15)
            #DrawBox(image, person_box, colour=(0,0,255), thickness=15)
            #print("IMPORTANT person_box: {}".format(person_box))
#Draw boxes for people - but if using slow version            
if doPersonDetection and slowDetection:            
    for box in parsed_boxes:
        DrawBox(image, box, colour=(0,0,255), thickness=20)
image.save("letsgo.JPEG", "JPEG")
Image.open("letsgo.JPEG").show()

        







