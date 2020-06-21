import numpy as np
import struct
import tensorflow as tf

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H +  padding - field_height) % stride == 0
    assert (W + padding - field_height) % stride == 0
    out_height = H
    out_width = W
    out_height = int(out_height)
    out_width = int(out_width)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p//2, (p+1)//2), (p//2, (p+1)//2)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def conv2d_py(X, kernel, output):
    h_filter, w_filter, d_filter, n_filters = kernel.shape
    W = kernel.transpose(3, 2, 0, 1)
    n_x, h_x, w_x, d_x = X.shape
    X = X.transpose(0, 3, 1, 2)
    padding = h_filter - 1

    h_out = h_x
    w_out = w_x
    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding, stride=1)
    W_col = W.reshape(n_filters, -1)

    out = W_col @ X_col
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 1, 2, 0)

    assert out.shape == output.shape
    n, h, w, c = out.shape
    for i in range(n):
        for j in range(h):
            for l in range(w):
                for t in range(c):
                    if abs(out[i,j,l,t] - output[i,j,l,t]) > 0.01:
                        print(out[i,j,l,t])
                        print(output[i,j,l,t])
                        print(abs(out[i,j,l,t] - output[i,j,l,t]))
                        print("Failed at: ",i ,j, l, t)
                        exit(-1)

def conv2d_tf(X, W, output):
    x = tf.constant(X, dtype=tf.float32)
    kernel = tf.constant(W, dtype=tf.float32)
    tensor = tf.nn.conv2d(
        x, kernel, 1, "SAME", data_format='NHWC', dilations=None, name=None
    )
    sess = tf.Session()
    out = sess.run(tensor)
    n, h, w, c = out.shape
    # print(out[0,0,0,0])
    print(np.mean(out))
    print("ERROR", np.sum(np.absolute(output - out)))
    for i in range(n):
        for j in range(h):
            for l in range(w):
                for t in range(c):
                    if abs(out[i,j,l,t] - output[i,j,l,t]) > 0.01:
                        print(out[i,j,l,t])
                        print(output[i,j,l,t])
                        print("Failed at: ",i ,j, l, t)
                        print(abs(out[i,j,l,t] - output[i,j,l,t]))
                        exit(-1)

    # print("ERROR", np.sum(np.absolute(output - out)))
    

def main():
    with open("/home/mirali/Conv2D/group2/2/input_tensor.bin", mode='rb') as file:
        fileContent = file.read()
    n, h, w, c = struct.unpack('iiii', fileContent[:16])
    input_img = struct.unpack("f" * ((len(fileContent) - 16) // 4), fileContent[16:])
    X = np.array(input_img)
    X = X.reshape(n, h, w, c)
    print(X.min(), X.max())
    print(np.mean(X))

    with open("/home/mirali/Conv2D/group2/2/kernel_tensor.bin", mode='rb') as file:
        fileContent = file.read()
    kh, kw, oc, ic = struct.unpack('iiii', fileContent[:16])
    kernel = struct.unpack("f" * ((len(fileContent) - 16) // 4), fileContent[16:])
    W = np.array(kernel)
    W = W.reshape(kh, kw, ic, oc)
    print(W.min(), W.max())
    print(np.mean(W))

    with open("/home/mirali/Conv2D/prob1/output_tensor.bin", mode='rb') as file:
        fileContent = file.read()
    on, oh, ow, out_c = struct.unpack('iiii', fileContent[:16])
    assert n==on
    assert oh == h, "oh = {}, h = {}".format(oh, h)
    assert ow == w
    assert out_c == oc
    output = struct.unpack("f" * ((len(fileContent) - 16) // 4), fileContent[16:])
    out = np.array(output)
    out = out.reshape(on, oh, ow, out_c)
    print(out.min(), out.max())

    # conv2d_py(X, W, out)
    conv2d_tf(X, W, out)
main()
