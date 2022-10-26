# import the necessary packages
import matplotlib.pyplot as plt
from google.colab import drive
import os
import tarfile
import cv2
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import BatchNormalization
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import shutil
import time

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

def TestMatPlotLib():
  x  = [1, 2, 3, 4, 5, 6, 7, 8, 9]
  y1 = [1, 3, 5, 3, 1, 3, 5, 3, 1]
  y2 = [2, 4, 6, 4, 2, 4, 6, 4, 2]
  plt.plot(x, y1, label="line L")
  plt.plot(x, y2, label="line H")
  plt.plot()

  plt.xlabel("x axis")
  plt.ylabel("y axis")
  plt.title("Line Graph Example")
  plt.legend()
  plt.show()
  return()

def UnTarShrink32(dirname, tfile):
  if os.path.isdir(dirname):
    print('[INFO] UnTarShrink32: ', dirname, 'folder already exists, doing nothing')
  else:
    os.mkdir(dirname)
    os.chdir(dirname)
    tar=tarfile.open(tfile)
    tar.extractall()
    tar.close()
  return()

# rd is read dir, wd is write dir
def CropShrink(rd, wd):
  if not os.path.isdir(wd):
    print('CropShrink: creating write dir ', wd)
    os.mkdir(wd)
  os.chdir('/content/gdrive/My Drive/Science Fair 2020/')
  fd=open("annotationposchknnorig.txt","r") 
  for line in fd:
    # skip the first 56 characters (old directory path)
    strin = line[56:len(line)]
    sstr = strin.split(' ')
    # splice in _p into filename
    fname = sstr[0]
    fsplit = fname.split('.')
    name = fsplit[0]+'_p'
    fname = name+'.JPG'
    readname = os.path.join(rd, fname)
    writename = os.path.join(wd, fname)
    # convert sstr into a numeric equivalent
    i=1
    nstr = []
    while i < len(sstr):
      nstr.append(int(sstr[i]))
      i=i+1
    image = cv2.imread(readname)
    crop_img = image[nstr[2]:(nstr[2]+nstr[4]), nstr[1]:(nstr[1]+nstr[3])]
    if ((nstr[3] == 0) or (nstr[4] == 0)):
      cv2.imwrite(writename, image)
    else: 
      cv2.imwrite(writename, crop_img)
    if nstr[0] >= 2:
      crop_img = image[nstr[6]:(nstr[6]+nstr[8]), nstr[5]:(nstr[5]+nstr[7])]
      cv2.imwrite(writename+'-2.JPG', crop_img)
    if nstr[0] >= 3:
      crop_img = image[nstr[10]:(nstr[10]+nstr[12]), nstr[9]:(nstr[9]+nstr[11])]
      cv2.imwrite(writename+'-3.JPG', crop_img)
    if nstr[0] == 4:
      crop_img = image[nstr[14]:(nstr[14]+nstr[16]), nstr[13]:(nstr[13]+nstr[15])]
      cv2.imwrite(writename+'-4.JPG',crop_img)
    if nstr[0] > 4:
      print('error - have more than 4 roaches in the annotations')
    # end of for loop
  fd.close()
  return()

class LeNet:
    @staticmethod
    def Build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
          inputShape = (depth, height, width)
        # first 
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape, activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second 
        model.add(Conv2D(50, (5, 5), padding="same", activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500, activation ='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

def TrainShrink(dirname, ver, NPOS, NNEG, IMGX, IMGY):
  # initialize the number of epochs to train for, initial learning rate,
  # and batch size
  EPOCHS = 25
  INIT_LR = 1e-3
  BS = 16

  # initialize the data and labels
  print("[INFO] loading images...")
  data = []
  labels = []

  # grab the image paths and randomly shuffle them
  allposimagePaths = sorted(list(paths.list_images(dirname,'_p')))
  allnegimagePaths = sorted(list(paths.list_images(dirname,'_n')))
  print('positive images = ', str(len(allposimagePaths)))
  print('negative images = ', str(len(allnegimagePaths)))
  posimagePaths = allposimagePaths[0:NPOS]
  negimagePaths = allnegimagePaths[0:NNEG]
  imagePaths = posimagePaths + negimagePaths
  random.seed(42)
  random.shuffle(imagePaths)

  # loop over the input images
  for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    if (image is None):
      print(imagePath)
    image = cv2.resize(image, (IMGX, IMGY))
    image = img_to_array(image)
    data.append(image)
    # update the labels list
    if imagePath in posimagePaths:
      label = 1
    else:
       label = 0
    labels.append(label)
  
  # scale the raw pixel intensities to the range [0, 1]
  data = np.array(data, dtype="float") / 255.0
  labels = np.array(labels)
  #print(sum(labels))
  #plt.plot(labels)
  print("data: " + str(data.shape))
  print("labels: " + str(labels.shape))

  # partition the data into training and testing splits using 75% of
  # the data for training and the remaining 25% for testing
  (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.5, random_state=42)

  # convert the labels from integers to vectors
  trainY = to_categorical(trainY, num_classes=2)
  testY = to_categorical(testY, num_classes=2)
  # construct the image generator for data augmentation
  aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
  # initialize the model
  print("[INFO] compiling model...")
  model = LeNet.Build(width=IMGX, height=IMGY, depth=3, classes=2)
  opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
  model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

  # train the network
  print("[INFO] training network...")
  H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS, verbose=1, shuffle=True)

  # save the model to disk
  print("[INFO] serializing network...")
  file='/content/gdrive/My Drive/Science Fair 2020/' + ver + '.model'
  model.save(file)

  # plot the training loss and accuracy
  plt.style.use("ggplot")
  plt.figure()
  N = EPOCHS
  plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
  plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
  plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy on Roach/Plate")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend(loc="lower left")
  file='/content/gdrive/My Drive/Science Fair 2020/' + ver + 'plt.png'
  plt.savefig(file)
  return(model)

def PredictShrink(dirname, model, IMGX, IMGY):
  imagePaths = sorted(os.listdir(dirname))
  print('making', len(imagePaths), 'predictions')
  print('total size:', len(imagePaths)*IMGX*IMGY)
  num = 0
  batch = 512
  # batched prediction loop to fit in memory
  all_preds = np.empty([len(imagePaths), 2], dtype=np.float32)  # pre-allocate required memory for array for efficiency
  batch_indices = np.arange(start=0, stop=len(imagePaths), step=batch)  # row indices of batches
  batch_indices = np.append(batch_indices, len(imagePaths))  # add final batch_end row
  print('batched as', batch_indices)
  for index in np.arange(len(batch_indices) - 1):
    batch_start = batch_indices[index]  # first row of the batch
    batch_end = batch_indices[index + 1]  # last row of the batch
    sub_paths = imagePaths[batch_start:batch_end]
    ndata = []
    for imagePath in imagePaths[batch_start:batch_end]:
      image = cv2.imread(os.path.join(dirname,imagePath))
      image = cv2.resize(image, (IMGX, IMGY))
      image = img_to_array(image)
      ndata.append(image)
    ndata = np.array(ndata, dtype="float") / 255.0
    #print(index, "ndata: " + str(ndata.shape))
    all_preds[batch_start:batch_end] = model.predict_on_batch(ndata)
  return(all_preds)

def SubCompute(prefix, preds, paths, allPaths):
  total = len(paths)
  misses = np.zeros(total)
  false_hits = np.zeros(total)
  pos = np.zeros(total)
  neg = np.zeros(total)
  for index in np.arange(total):
    pred_index = allPaths.index(paths[index])
    if '_p' in paths[index]:
      pos[index] = 1
      if (preds[pred_index,1] < 0.5):
        misses[index] = 1
    elif '_n' in paths[index]:
      neg[index] = 1
      if (preds[pred_index,1] >= 0.5):
        false_hits[index] = 1
    else:
      print('computeMissesAndFalseHits error: filename without _p or _n', index)
  count_misses = sum(misses)
  count_false_hits = sum(false_hits)
  metric = 1 - 0.5*(count_misses/sum(pos)) - 0.5*(count_false_hits/sum(neg))
  print(prefix, end='')
  print(': pos', sum(pos), 'neg', sum(neg), 'misses', count_misses, 'false_hits', count_false_hits, 'metric', str.format("{0:.3f}",metric))
  return(misses, false_hits)

def computeMissesAndFalseHits(dirname, preds, NPOS, NNEG):
  imagePaths = sorted(os.listdir(dirname))
  total = len(imagePaths)
  allposimagePaths = sorted(list(paths.list_images(dirname, '_p')))
  allnegimagePaths = sorted(list(paths.list_images(dirname, '_n')))
  # strip directory from paths
  allposimagePaths = [os.path.basename(path) for path in allposimagePaths]
  allnegimagePaths = [os.path.basename(path) for path in allnegimagePaths]
  trainimagePaths = sorted(allposimagePaths[0:NPOS] + allnegimagePaths[0:NNEG])
  #TNPOS = min(439,3*NPOS)
  #TNNEG = min(23378,10*NNEG)
  #unusedimagePaths = sorted(allposimagePaths[NPOS:TNPOS]+allnegimagePaths[NNEG:TNNEG])
  #restimagePaths = sorted(allposimagePaths[TNPOS:len(allposimagePaths)]+allnegimagePaths[TNNEG:len(allnegimagePaths)])
  validationimagePaths = sorted(allposimagePaths[NPOS:len(allposimagePaths)]+allnegimagePaths[NNEG:len(allnegimagePaths)])
  train_misses, train_false_hits = SubCompute('train', preds, trainimagePaths, imagePaths)
  #unused_misses, unused_false_hits = SubCompute('unused', preds, unusedimagePaths, imagePaths)
  #rest_misses, rest_false_hits = SubCompute('rest', preds, restimagePaths, imagePaths)
  validation_misses, validation_false_hits = SubCompute('validation', preds, validationimagePaths, imagePaths)
  misses, false_hits = SubCompute('all', preds, imagePaths, imagePaths)
  return(misses, false_hits)

def DebugMissesAndFalseHits(refdirname, dirname, preds, misses, false_hits, IMGX, IMGY, moviefnamep, moviefnamen):
  imagePaths = sorted(os.listdir(dirname))
  imagefns = []
  for index in np.arange(len(imagePaths)):
    if misses[index] == 1:
      imagefns.append(imagePaths[index])
  imagesToMovie(refdirname, dirname, imagePaths, imagefns, moviefnamep, IMGX, IMGY, preds)
  imagefns = []
  for index in np.arange(len(imagePaths)):
    if false_hits[index] == 1:
      imagefns.append(imagePaths[index])
  imagesToMovie(refdirname, dirname, imagePaths, imagefns, moviefnamen, IMGX, IMGY, preds)
  return()

# generate subtracted images
# note the Combine version is the same as the non Combine version
def GenShrinkSubImage(dirname):
  os.chdir(dirname)
  imagefns = sorted(os.listdir())
  lastim = ''
  lastfns10 = imagefns[0].split('_')[0]
  lastfns11 = imagefns[0].split('_')[1]
  i = 0
  for imagefn in imagefns:
    im = cv2.imread(imagefn)
    fns1=imagefn.split('_')
    if len(lastim) == 0:
      # start: subtract from self
      sub = cv2.subtract(im, im)
    else:
      if lastfns10 == fns1[0]:
        # subtract from prev image
        sub = cv2.subtract(im, lastim)
      else:
        # reset batch
        sub = cv2.subtract(im, im)
    lastim = im
    lastfns10 = fns1[0]
    cv2.imwrite(imagefn, sub)
  return()

def cmpandpltimages(imfn1, imfn2, plot):
  im1 = cv2.imread(imfn1)
  im2 = cv2.imread(imfn2)
  sub=cv2.subtract(im1, im2)
  if plot:
    plt.imshow(np.concatenate((im1, im2, sub), axis=1))
    #plt.show()
  return(sub)

def imagesToMovie(refDir, imDir, imagePaths, imagefns, moviefname, IMGX, IMGY, preds):
  font = cv2.FONT_HERSHEY_SIMPLEX
  #these were tried with 29.44
  fourcc = cv2.VideoWriter_fourcc(*"MJPG")
  video = cv2.VideoWriter(moviefname, fourcc, 15, (3*IMGX, IMGY))
  print(moviefname, ' opened ?', video.isOpened())
  # Appending the images to the video one by one
  im_i = 0
  for imagefn in imagefns:
    refImageIndex = imagePaths.index(imagefn)
    imRefLfname = imagePaths[refImageIndex-1] #L for last
    im = cv2.imread(os.path.join(imDir, imagefn))
    imRefL = cv2.imread(os.path.join(refDir, imRefLfname))
    imRefC = cv2.imread(os.path.join(refDir, imagefn)) # c for Current
    #edges = cv2.Canny(im, 10, 100)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #dilated = cv2.dilate(edges,kernel,iterations = 1)
    #contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #for i,contour in enumerate(contours):
      #area = cv2.contourArea(contour)
      #if area > 1000.0:
        #cv2.drawContours(im, contours, i, (0,255,255), 2)
    fn = os.path.basename(imagefn)
    fnparts = fn.split('-')
    fn1 = fnparts[1]
    fn2 = fnparts[2]
    fn = fn1[2:6] + '-' + fn2[4:8]
    fn = imagefn[11:15] + '-' + imagefn[20:24]
    cv2.putText(im, str(preds[refImageIndex]), (50,50), font, 1, (255,255,255), 2, cv2.LINE_AA)
    fnL = imRefLfname[11:15] + '-' + imRefLfname[20:24]
    cv2.putText(imRefL, fnL, (50,50), font, 1, (255,255,255), 2, cv2.LINE_AA)
     #fn = imRefCfname[11:15] + '-' + imRefCfname[20:24]
    cv2.putText(imRefC, fn, (50,50), font, 1, (255,255,255), 2, cv2.LINE_AA)
    im = cv2.resize(im, (IMGX, IMGY))
    imRefL = cv2.resize(imRefL, (IMGX, IMGY))
    imRefC = cv2.resize(imRefC, (IMGX, IMGY))
    im = np.concatenate((imRefL, im, imRefC), axis=1)
    video.write(im)
    im_i = im_i+1
  video.release() # releasing the video generated
  return()





















