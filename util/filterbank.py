import numpy as np


def gaussian1d(sigma, mean, x, order):
    x = np.array(x)
    x_ = x - mean
    var = np.power(sigma, 2)

    gaussian = (1 / np.sqrt(2 * np.pi * var)) * (np.exp((-1 * x_ * x_) / (2 * var)))
    if order == 0:
        result = gaussian
        return result
    elif order == 1:
        result = -gaussian * (x_ / var)
        return result
    else:
        result = gaussian * ((x_ * x_ - var) / np.power(var, 2))
        return result


def gaussian2d(sup, scales):
    var = scales * scales
    shape = (sup, sup)
    m, n = [(i - 1) / 2 for i in shape]
    x, y = np.ogrid[-m:m + 1, -n:n + 1]
    gaussian = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(np.power(x, 2) + np.power(y, 2) / (2 * var)))
    return gaussian


def log2d(sup, scales):
    var = scales * scales
    shape = (sup, sup)
    m, n = [(i - 1) / 2 for i in shape]
    x, y = np.ogrid[-m:m + 1, -n:n + 1]
    gaussian = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(np.power(x, 2) + np.power(y, 2) / (2 * var)))
    h = gaussian * ((np.power(x, 2) + np.power(y, 2)) / np.power(var, 2))
    return h


def makefilter(scale, phasex, phasey, pts, sup):
    gx = gaussian1d(3 * scale, 0, pts[0, ...], phasex)
    gy = gaussian1d(scale, 0, pts[1, ...], phasey)

    image = gx * gy
    image = np.reshape(image, (sup, sup))
    return image


def makeLMfilters():
    sup = 5
    scalex = np.sqrt(2) * np.array([1, 2, 3])
    norient = 6
    nrotinv = 12

    nbar = len(scalex) * norient
    nedge = len(scalex) * norient
    nf = nbar + nedge + nrotinv
    F = np.zeros([sup, sup, nf])
    hsup = (sup - 1) / 2
    x = [np.arange(-hsup, hsup + 1)]
    y = [np.arange(-hsup, hsup + 1)]

    [x, y] = np.meshgrid(x, y)
    orgpts = [x.flatten(), y.flatten()]
    orgpts = np.array(orgpts)

    count = 0

    for scale in range(len(scalex)):
        for orient in range(norient):
            angle = (np.pi * orient) / norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotpts = [[c + 0, -s + 0], [s + 0, c + 0]]
            rotpts = np.array(rotpts)
            rotpts = np.dot(rotpts, orgpts)
            F[:, :, count] = makefilter(scalex[scale], 0, 1, rotpts, sup)
            F[:, :, count + nedge] = makefilter(scalex[scale], 0, 2, rotpts, sup)
            count = count + 1

    count = nbar + nedge
    scales = np.sqrt(2) * np.array([1, 2, 3, 4])

    for i in range(len(scales)):
        F[:, :, count] = gaussian2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:, :, count] = log2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:, :, count] = log2d(sup, 3 * scales[i])
        count = count + 1

    return F


"""
F = makeLMfilters()

print(F.shape)
for i in range(0, 18):
    plt.subplot(3, 6, i + 1)
    plt.axis('off')
    plt.imshow(F[:, :, i], cmap='gray')
plt.show()
for i in range(0,18):
    plt.subplot(3,6,i+1)
    plt.axis('off')
    plt.imshow(F[:,:,i+18], cmap = 'gray')

plt.show()
for i in range(0,12):
    plt.subplot(4,4,i+1)
    plt.axis('off')
    plt.imshow(F[:,:,i+36], cmap = 'gray')

plt.show()
"""
