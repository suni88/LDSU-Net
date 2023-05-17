from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/train','image','mask',data_gen_args,save_to_dir = None)

model = udnet()
model_checkpoint = ModelCheckpoint('LDSU-Net.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit(myGene,steps_per_epoch=200,epochs=10,callbacks=[model_checkpoint])

testGene = testGenerator("data/test")
results = model.predict(testGene,2,verbose=1)
saveResult("data/test",results)
