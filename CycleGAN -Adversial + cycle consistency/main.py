import tensorflow as tf
# import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from IPython import display
import pickle
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
import tensorflow_addons as tfa
from matplotlib import pyplot
from tensorflow.keras.initializers import RandomNormal

plt.ion()
def plot(a, b,c,d):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(a)
    plt.plot(b)
    plt.plot(c)
    plt.plot(d)
    plt.ylim(ymin=0)
    plt.show(block=False)
    plt.pause(.1)

def createdis():
    # PatchGan
    init = RandomNormal(stddev=0.02)
    dis = Sequential([
        layers.Conv2D(64, kernel_size=4, strides=2,padding='same',kernel_initializer=init),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=4, strides=2,padding='same',kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(256, kernel_size=4, strides=2,padding='same',kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(512, kernel_size=4, strides=2,padding='same',kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(512, kernel_size=4,padding='same',kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init)
    ])
    dis.compile(loss='mse', optimizer=Adam(learning_rate=0.00002, beta_1=0.75), loss_weights=[0.8])#added 1 more 0 to learning rate
    return dis
OtherDiscrim = createdis()
MeDiscrim = createdis()

def create_gen():
    init = RandomNormal(stddev=0.02)
    return Sequential([
        # ---------inital
        layers.Conv2D(64, kernel_size=7,padding='same',kernel_initializer=init),
        layers.ReLU(0.2),
        # ----------Down
        layers.Conv2D(64 * 2, kernel_size=3, strides=2,padding='same',kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform"),
        layers.ReLU(0.2),
        layers.Conv2D(64 * 4, kernel_size=3, strides=2,padding='same',kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform"),
        layers.ReLU(0.2),
        # ----------Residuals
        layers.Conv2D(64 * 4, kernel_size=3,padding='same',kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),
        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),

        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),
        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),

        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),
        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),

        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),
        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),

        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),
        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),

        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),
        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),

        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),
        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),

        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),
        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),

        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),
        layers.Conv2D(64 * 4, kernel_size=3, padding='same', kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),


        # -----------Upsampling
        layers.Conv2DTranspose(64 * 2, kernel_size=3, strides=2,padding='same',kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",gamma_initializer="random_uniform"),
        layers.ReLU(0.2),
        layers.Conv2DTranspose(64, kernel_size=3, strides=2,padding='same',kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform"),
        layers.ReLU(0.2),
        layers.Conv2D(3, kernel_size=7,padding='same',kernel_initializer=init),
        tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform"),
        layers.Activation('tanh')
    ])

MeGen = create_gen()
OtherGen = create_gen()




def define_composite_model(generator, discriminator, generatorTwo, image_shape):
    generator.trainable = True
    discriminator.trainable = False
    generatorTwo.trainable = False
    input_gen = layers.Input(shape=image_shape)
    gen1_out = generator(input_gen)
    output_d = discriminator(gen1_out)
    output_f = generatorTwo(gen1_out)

    model = Model([input_gen], [output_d, output_f])
    model.compile(loss=['mse', 'mae'], loss_weights=[5,10], optimizer=Adam(learning_rate = 0.002))
    return model



Me_To_Other = define_composite_model(OtherGen, OtherDiscrim, MeGen, (256,256,3))
Other_To_Me = define_composite_model( MeGen, MeDiscrim, OtherGen, (256,256,3))


def generate_real_samples(dataset, n_samples, patch_size):
    ind = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ind]
    y = np.ones((n_samples, patch_size, patch_size, 1))
    return X, y

def generate_fake_samples(generator, dataset, img_shape):
    X = generator.predict(dataset)
    y = np.zeros((len(X), img_shape, img_shape, 1))
    return X, y

def save_models(step, meToBidenGen, BidenTomeGen):
    meToBidenGen.save('D:/CycleGanModels/OtherToMe%06d.h5' % (step+1))
    BidenTomeGen.save('D:/CycleGanModels/MeToOther%06d.h5' % (step+1))

def summarize_performance(step, g_model, trainX, name, n_samples=5):
    X_in = generate_real_samples(trainX, n_samples, 0)[0]
    X_out = generate_fake_samples(g_model, X_in, 0)[0]
    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_in[i])
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_out[i])
    pyplot.savefig('D:/CycleGanPics/%s_generated_plot_%06d.png' % (name, (step+1)))
    pyplot.close()


def train(MeDiscriminator, BidenDiscriminator, BidenGenerator, MeGenerator, MeToBidenComp, BidenToMeComp, dataset):
    epochs = 100
    patch_size = MeDiscriminator.output_shape[1]
    trainA, trainB = dataset
    n_steps = epochs*(len(trainA))

    batch_size = 1
    meToBidenGenLoss = []
    BidentomeGenLoss = []
    BidenDiscrimloss = []
    MeDiscrimloss = []
    for i in range(n_steps):
        X_realA, y_realA = generate_real_samples(trainA, batch_size, patch_size)
        X_realB, y_realB = generate_real_samples(trainB, batch_size, patch_size)
        X_fakeA, y_fakeA = generate_fake_samples(MeGenerator, X_realB, patch_size)
        X_fakeB, y_fakeB = generate_fake_samples(BidenGenerator, X_realA, patch_size)

        g_loss2 = BidenToMeComp.train_on_batch([X_realB], [y_realA, X_realB])[0]
        BidentomeGenLoss.append(g_loss2)
        dA_loss1 = MeDiscriminator.train_on_batch(X_realA, y_realA)
        dA_loss2 = MeDiscriminator.train_on_batch(X_fakeA, y_fakeA)
        MeDiscrimloss.append((dA_loss1+dA_loss2)/2)
        g_loss1 = MeToBidenComp.train_on_batch([X_realA], [y_realB, X_realA])[0]
        meToBidenGenLoss.append(g_loss1)
        dB_loss1 = BidenDiscriminator.train_on_batch(X_realB, y_realB)
        dB_loss2 = BidenDiscriminator.train_on_batch(X_fakeB, y_fakeB)
        BidenDiscrimloss.append((dB_loss1+dB_loss2)/2)
        print("Me Discrim loss-",(dA_loss1+dA_loss2)/2,":::Biden Discrim loss-",(dB_loss1+dB_loss2)/2,"\nMe Gen loss-",g_loss1,":::Biden gen loss-",g_loss2)
        plot(meToBidenGenLoss, BidentomeGenLoss, BidenDiscrimloss, MeDiscrimloss)
        if (i+1) % 2500 == 0:
            summarize_performance(i, BidenGenerator, trainA, 'MeToOther')
            summarize_performance(i, MeGenerator, trainB, 'OtherToMe')
        if (i+1) % 5000 == 0:
            save_models(i, BidenGenerator, MeGenerator)



dataset = [pickle.load( open("mypicsfinal.p", "rb" ) ) ,pickle.load( open("bidenpicsfinal.p", "rb" ) )]
train(MeDiscrim, OtherDiscrim, OtherGen, MeGen, Me_To_Other, Other_To_Me, dataset)