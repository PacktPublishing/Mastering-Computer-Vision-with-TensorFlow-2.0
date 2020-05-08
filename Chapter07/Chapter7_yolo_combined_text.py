import glob, os
# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
#current_dir = '/home/krishkar/Documents/chapter7_yolo/furniture_data/trainyolo'
current_dir = '/home/krishkar/Documents/chapter7_yolo/furniture_data/testyolo'
# Create and/or truncate train.txt and test.txt
#file_train = open('train.txt', 'w')  
file_test = open('test.txt', 'w')
counter = 1  
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
   # file_train.write(current_dir + "/" + title + '.jpg' + "\n")
    file_test.write(current_dir + "/" + title + '.jpg' + "\n")
    counter = counter + 1
