# choix de la zone selon disponibilité :
BUCKET_NAME = "projet8-oc"   # eu-west-1
# BUCKET_NAME = "oc-projet-8"   # eu-west-3

# imports modules
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from pyspark import SparkContext
import boto3
from datetime import datetime
from time import time

def timestamp():
    return datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

def logger_s3(s3_bucket, logfile, log, log_to_shell=True):
    """Appends log to logfile. Uploads each update to S3 bucket.
    Args :
        - s3_bucket (Bucket).
        - logfile (string) .
        - log (string).
        - log_to_shell (bool) : si True, imprime également dans le std.out.
    Returns : /
    """
    time_log = timestamp() + " : " + log + "\n"
    if log_to_shell : print("*"*100 + "\n", time_log)
    with open("./logs/"+ logfile, 'a') as f:
        f.write(time_log)
    s3_bucket.upload_file("./logs/"+ logfile, 'logs/' + logfile)

# fonction d'extraction de feature
def VGG16_extracteur_spark(path, nom_image, model):
    """Transforme un fichier image en un vecteur de dimension 4096.
    Args :
    - path : chemin vers les images (chemin local ou 'S3').
    - nom_image : chemin d'accès à l'image (exemple : 'Apple Braeburn/r_173_100.jpg')
    - model : model_vgg16 ou bc_model_vgg16.
    Returns :
    - liste de dimension 4097 (4096 dimensions de VGG16 + nom_image).
    """
    # create path to the image stored locally
    if path == 'S3':
        s3_bucket_vgg16 = boto3.resource('s3').Bucket(BUCKET_NAME)    # il faut un une instance Bucket différente sur le driver et les exécuteurs
        s3_bucket_vgg16.download_file('input/' + nom_image, '/tmp/img')
        path_nom_image = '/tmp/img'
    else:
        path_nom_image = os.path.join(path, nom_image)
    # load the image for keras processing
    image = load_img(path_nom_image, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get extracted features
    im_features = model.predict(image)
    # convert to list and add nom_image
    im_features = im_features[0].tolist()
    im_features.append(nom_image)
    return im_features

# il faut un une instance Bucket différente sur le driver et les exécuteurs
s3_bucket = boto3.resource('s3').Bucket(BUCKET_NAME)

# start logging
os.makedirs(os.path.join(os.getcwd(), 'logs/'), exist_ok=True)
logfile = "Log for job started " + timestamp() + ".txt"
logger_s3(s3_bucket, logfile, "logging starts")

# liste des fruits dans le répertoire "input/" : megabatch_img
all_fruits = s3_bucket.objects.filter(Prefix="input")
megabatch_img = []
for obj in all_fruits:
    megabatch_img.append(obj.key.lstrip("input/"))
del megabatch_img[0]  # suppression d'un objet non pertinent

# load model VGG16
model_vgg16 = VGG16()
# remove the output layer
model_vgg16 = Model(inputs=model_vgg16.inputs, outputs=model_vgg16.layers[-2].output)

# Spark context : le SparkConf est supprimé car géré via le spark-submit
sc = SparkContext()

# logger des paramètres du spark context lancé
logger_s3(s3_bucket, logfile, "Configuration de Spark :" + str(sc.getConf().getAll()))

# broadcasting
bc_model_vgg16 = sc.broadcast(model_vgg16)

# logging
logger_s3(s3_bucket, logfile, "spark job starts")
t0 = time()

# job spark
resultat = sc.parallelize(megabatch_img) \
    .map(lambda img: VGG16_extracteur_spark('S3', img, bc_model_vgg16.value)) \
    .collect()

# logging
logger_s3(s3_bucket, logfile, f"spark job has ended (duration of {round(time() - t0, 1)}s) - output to S3 starts")

# output
df_output = pd.DataFrame(resultat)
df_output.columns = [f'dim_{i}' for i in range(4096)] + ['path']
file_name = f'df_output_{len(df_output)}_fruits.csv'
local_path = os.path.join(os.getcwd(), file_name)
df_output.to_csv(path_or_buf=local_path)
s3_bucket.upload_file(local_path, 'output/' + file_name)

# end logging
logger_s3(s3_bucket, logfile, "output to S3 has ended")
