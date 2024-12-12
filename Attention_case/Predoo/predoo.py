import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import time
from queue import Queue
import math
a = []
a1 = 0.0001 * np.ones((1, 3, 10, 10), dtype=np.float64)
a2 = 0.000001 * np.ones((1, 3, 10, 10), dtype=np.float64)
a3 = 0.00000001 * np.ones((1, 3, 10, 10), dtype=np.float64)

def input_withDiffDype(x, dtype):
    return torch.tensor(x, dtype=dtype).cuda()

#def torch_convWithDiffDype(dtype):
    #return nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=0, dtype=dtype).double().cuda()
#    return nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=0,dtype=dtype).cuda()
    #torch con2v卷积核内权重要保持一致
    #return nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=0).cuda()



def createCorpus(n):
    q = Queue()
    for i in range(n):
        x = np.random.randn(1, 3, 10, 10)
        q.put(x)
    return q


def Max_guided(corpus, f, g , s , s2, c1 ,c2):
    out = open(file=f, mode='a', newline='')
    csv_writer = csv.writer(out)
    out1 = open(file=g, mode="a", newline='')
    csv_writer1 = csv.writer(out1)
    csv_writer.writerow(["No.", "16_32(16)", "16_64(16)", "32_16(32)", "32_64(32)", "64_16(64)", "64_32(64)",
                         "time1", "32_16(16)", "64_16(16)", "16_32(32)", "64_32(32)", "16_64(64)", "32_64(64)", "time2",
                         "isNaN"])
    csv_writer1.writerow(
        ["No.", "当前最大误差(同输入)", "全局最大误差(同输入)", "引起最大误差的输入编号1", "当前最大误差(同算子)",
         "全局最大误差(同算子)", "引起最大误差的输入编号2"])
    h_error1 = 0
    h_error2 = 0
    maxine1 = 0
    maxine2 = 0
    j = 0
    index1 = 0
    index2 = 0
    while not corpus.empty() and j < 20000:
        x = corpus.get()
        y = corpus.get()
        z = corpus.get()
        # 为了存储con2d的权重：
        #con16 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=0, dtype=torch.float16).cuda()
        maxse, maxe1, maxe2 = getMaxdiff(x, y,z, csv_writer, j)
        if maxe1 > maxine1:
            torch.save(x, s)
            torch.save(y, c1)
            torch.save(x, c2)
            #torch.save(con16, c1)
            index1 = j
            maxine1 = maxe1# 最大误差
        if maxe2 > maxine2:
            index2 = j
            #torch.save(x, s2)
            #torch.save(con16, c2)
            maxine2 = maxe2  # 最大误差
        if maxse > 0.0025:
            corpus.put(x + a1)
            corpus.put(y + a2)
            corpus.put(z + a3)
        if j % 999 == 0:
            r = []
            h_error1 = max(h_error1, maxine1)
            h_error2 = max(h_error2, maxine2)
            r.append(j // 999)
            r.append(maxine1)
            r.append(h_error1)
            r.append(index1)
            r.append(maxine2)
            r.append(h_error2)
            r.append(index2)
            csv_writer1.writerow(r)
            #maxine1 = 0
            #maxine2 = 0
            index1 = 0
            index2 = 0
        j += 1
        print(j)
    out.close()
    out1.close()

'''
def Mean_guided(corpus, f, g):
    out = open(file=f, mode='a', newline='')
    csv_writer = csv.writer(out)
    out1 = open(file=g, mode="a", newline='')
    csv_writer1 = csv.writer(out1)
    csv_writer.writerow(["No.", "16_32(16)", "16_64(16)", "32_16(32)", "32_64(32)", "64_16(64)", "64_32(64)",
                         "time1", "32_16(16)", "64_16(16)", "16_32(32)", "64_32(32)", "16_64(64)", "32_64(64)", "time2",
                         "isNaN"])
    csv_writer1.writerow(
        ["No.", "当前最大误差(同输入)", "全局最大误差(同输入)", "引起最大误差的输入编号1", "当前最大误差(同算子)",
         "全局最大误差(同算子)", "引起最大误差的输入编号2"])
    h_error1 = 0
    h_error2 = 0
    maxine1 = 0
    maxine2 = 0
    j = 0
    index1 = 0
    index2 = 0
    while not corpus.empty() and j < 20000:
        x = corpus.get()
        maxe1, maxe2 = getMeandiff(x, csv_writer, j)

        if max(maxe1, maxe2) > 5e-4:
            corpus.put(x + a1)
            corpus.put(x + a2)
            corpus.put(x + a3)

        if maxe1 > maxine1:
            index1 = j
            maxine1 = maxe1  # 最大误差

        if maxe2 > maxine2:
            index2 = j
            maxine2 = maxe2  # 最大误差

        if j % 999 == 0:
            r = []
            h_error1 = max(h_error1, maxine1)
            h_error2 = max(h_error2, maxine2)
            r.append(j // 999)
            r.append(maxine1)
            r.append(h_error1)
            r.append(index1)
            r.append(maxine2)
            r.append(h_error2)
            r.append(index2)
            csv_writer1.writerow(r)
            maxine1 = 0
            maxine2 = 0
            index1 = 0
            index2 = 0
        j += 1
        print(j)

    out.close()
    out1.close()
'''

def getMaxdiff(x,y,z, csv_writer, j, con16=None):
    res = []
    maxe = []
    res.append(j)
    x_32 = input_withDiffDype(x, torch.float32)
    x_16 = input_withDiffDype(x, torch.float16)
    x_64 = input_withDiffDype(x, torch.float64).double()
    y_32 = input_withDiffDype(y, torch.float32)
    y_16 = input_withDiffDype(y, torch.float16)
    y_64 = input_withDiffDype(y, torch.float64).double()
    z_32 = input_withDiffDype(z, torch.float32)
    z_16 = input_withDiffDype(z, torch.float16)
    z_64 = input_withDiffDype(z, torch.float64).double()
    s = time.time()
    torch_conv_16=torch.nn.functional.scaled_dot_product_attention
    torch_conv_32 =torch.nn.functional.scaled_dot_product_attention
    torch_conv_64 =torch.nn.functional.scaled_dot_product_attention
    out_16_16_1 = torch_conv_16(x_16,y_16,z_16).float()
    out_16_16_2 = torch_conv_16(x_16,y_16,z_16).double()
    #out_16_32 = torch_conv_32(x_16)
    #out_16_64 = torch_conv_64(x_16)
    #dif1 = torch.max(torch.abs(out_16_32 - out_16_16_1)).item()
    #dif2 = torch.max(torch.abs(out_16_64 - out_16_16_2)).item()
    out_32_32_1 = torch_conv_32(x_32,y_32,z_32)
    out_32_32_2 = torch_conv_32(x_32,y_32,z_32).double()
    #out_32_16 = torch_conv_16(x_32).float()
    #out_32_64 = torch_conv_64(x_32)
    diff1 = torch.max(torch.abs(out_32_32_1 - out_16_16_1)).item()
    diff2 = torch.max(torch.abs(out_32_32_1 - out_16_16_2)).item()
    diff3 = torch.max(torch.abs(out_32_32_2 - out_16_16_1)).item()
    diff4 = torch.max(torch.abs(out_32_32_2 - out_16_16_2)).item()
    #dif3 = torch.max(torch.abs(out_32_16 - out_32_32_1)).item()
    #dif4 = torch.max(torch.abs(out_32_64 - out_32_32_2)).item()
    #out_64_16 = torch_conv_16(x_64).double()
    #out_64_32 = torch_conv_32(x_64).double()
    out_64_64 = torch_conv_64(x_64,y_64,z_64)
    diff5 = torch.max(torch.abs(out_64_64 - out_16_16_2)).item()
    diff6 = torch.max(torch.abs(out_64_64 - out_32_32_2)).item()
    #dif5 = torch.max(torch.abs(out_64_16 - out_64_64)).item()
    #dif6 = torch.max(torch.abs(out_64_32 - out_64_64)).item()
    e = time.time()
    res.append(diff1)
    res.append(diff2)
    res.append(diff3)
    res.append(diff4)
    res.append(diff5)
    res.append(diff6)
    res.append(e - s)
    s = time.time()
    #for n in out_32_32_1.numpy().ravel():
    #    if math.isnan(n):
    #        res.append("NAN")
    #        break
    maxe.append(diff1)
    maxe.append(diff2)
    maxe.append(diff3)
    maxe.append(diff4)
    maxe.append(diff5)
    maxe.append(diff6)
    csv_writer.writerow(res)
    return max(maxe[:]), max(res[1:-1]), max(res[1:-1])


'''
def getMeandiff(x, csv_writer, j):
    res = []
    res.append(j)

    # Assuming 'x' is a NumPy array, convert it to a PyTorch tensor and move it to the GPU
    x_32 = torch.tensor(x, dtype=torch.float32).cuda()
    x_16 = torch.tensor(x, dtype=torch.float16).cuda()
    x_64 = torch.tensor(x, dtype=torch.float64).cuda()

    s = time.time()

    # Create convolutional layers for different data types
    torch_conv_16 = torch_convWithDiffDype('float16').cuda()
    torch_conv_32 = torch_convWithDiffDype('float32').cuda()
    torch_conv_64 = torch_convWithDiffDype('float64').cuda()

    out_16_16_1 = torch_conv_16(x_16).float().cpu().numpy()
    out_16_16_2 = torch_conv_16(x_16).double().cpu().numpy()
    out_16_32 = torch_conv_32(x_16)
    out_16_64 = torch_conv_64(x_16)

    # Compute mean absolute differences
    diff1 = torch.mean(torch.abs(out_16_32 - out_16_16_1)).item()
    diff2 = torch.mean(torch.abs(out_16_64 - out_16_16_2)).item()

    out_32_32_1 = torch_conv_32(x_32).float().cpu().numpy()
    out_32_32_2 = torch_conv_32(x_32).double().cpu().numpy()
    out_32_16 = torch_conv_16(x_32).float().cpu().numpy()
    out_32_64 = torch_conv_64(x_32)

    diff3 = torch.mean(torch.abs(out_32_16 - out_32_32_1)).item()
    diff4 = torch.mean(torch.abs(out_32_64 - out_32_32_2)).item()

    out_64_16 = torch_conv_16(x_64).double().cpu().numpy()
    out_64_32 = torch_conv_32(x_64).double().cpu().numpy()
    out_64_64 = torch_conv_64(x_64)

    diff5 = torch.mean(torch.abs(out_64_16 - out_64_64)).item()
    diff6 = torch.mean(torch.abs(out_64_32 - out_64_64)).item()
    e = time.time()

    res.append(diff1)
    res.append(diff2)
    res.append(diff3)
    res.append(diff4)
    res.append(diff5)
    res.append(diff6)
    res.append(e - s)

    s = time.time()
    out_16_16 = torch_conv_16(x_16).float().cpu().numpy()
    out_32_16_1 = torch_conv_16(x_32).float().cpu().numpy()
    out_64_16_1 = torch_conv_16(x_64).double().cpu().numpy()
    diff7 = torch.mean(torch.abs(out_32_16_1 - out_16_16)).item()
    diff8 = torch.mean(torch.abs(out_64_16_1 - out_16_16)).item()

    dif7 = torch.max(torch.abs(out_32_16_1 - out_16_16)).item()
    dif8 = torch.max(torch.abs(out_64_16_1 - out_16_16)).item()

    out_64_32_1 = torch_conv_32(x_64).double().cpu().numpy()
    diff9 = torch.mean(torch.abs(out_16_32 - out_32_32_1)).item()
    diff10 = torch.mean(torch.abs(out_64_32_1 - out_32_32_1)).item()

    dif9 = torch.max(torch.abs(out_16_32 - out_32_32_1)).item()
    dif10 = torch.max(torch.abs(out_64_32_1 - out_32_32_1)).item()

    diff11 = torch.mean(torch.abs(out_16_64 - out_64_64)).item()
    diff12 = torch.mean(torch.abs(out_32_64 - out_64_64)).item()

    dif11 = torch.max(torch.abs(out_16_64 - out_64_64)).item()
    dif12 = torch.max(torch.abs(out_32_64 - out_64_64)).item()
    e = time.time()

    res.append(diff7)
    res.append(diff8)
    res.append(diff9)
    res.append(diff10)
    res.append(diff11)
    res.append(diff12)
    res.append(e - s)

    for n in out_32_32_1.flatten():
        if math.isnan(n):
            res.append("NAN")
            break

    csv_writer.writerow(res)
    return max(res[1:7]), max(res[8:14])
'''


if __name__=='__main__':
    corpus=createCorpus(60000)
    # Max_guided(corpus,"E:\Dtype_test\Max_guided2\\tf_cpu_2.0.0\\tf_conv2d.csv","E:\Dtype_test\Max_guided2\\tf_cpu_2.0.0\\tf_conv2d_count.csv")
    # Mean_guided(corpus,"E:\Dtype_test\Mean_guided2\\tf_cpu_2.0.0\\tf_conv2d.csv","E:\Dtype_test\Mean_guided2\\tf_cpu_2.0.0\\tf_conv2d_count.csv")
    Max_guided(corpus,"C:/Users/14771/Desktop/5/FuzzTesting/paper/12sdpa.csv","C:/Users/14771/Desktop/5/FuzzTesting/paper/12sdpa_count.csv", 'C:/Users/14771/Desktop/5/FuzzTesting/paper/12sdpa.pt','C:/Users/14771/Desktop/5/FuzzTesting/paper/12sdpa.pt','C:/Users/14771/Desktop/5/FuzzTesting/paper/12sdpal.pt','C:/Users/14771/Desktop/5/FuzzTesting/paper/12sdpal2.pt',)
    #Mean_guided(corpus,"/home/ise/opTest/data/Mean_guided2/tf_gpu_2.0.0/conv2d.csv","/home/ise/opTest/data/Mean_guided2/tf_gpu_2.0.0/conv2d_count.csv")

