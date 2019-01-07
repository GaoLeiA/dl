import collections
import math
import os
import random
import zipfile
import urllib
import numpy as np
import tensorflow as tf

#定义下载文本数据的函数
# url = 'http://mattmahoney.net/dc/'
#
# def maybe_download(filename,expected_bytes):
#     if not os.path.exists(filename):
#         filename,_ = urllib.request.urlretrieve(url + filename,filename)
#     statinfo = os.stat(filename)  #访问一个文件的详细信息。
#     if statinfo.st_size == expected_bytes:  #文件大小(以字节为单位)
#         print('Found and verified(验证）',filename)
#     else:
#         print(statinfo.st_size)
#         raise Exception('Failed to verify(验证）' + filename + 'Can you get to it with a browser(浏览器)?')
#     return filename
#
# filename = maybe_download('text8.zip',31344016)

filename = './text8.zip'

#解压文件，并将数据转化成单词的列表
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        #获得名字列表，读取成字符串，编码成'utf-8'，最后进行分割
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
# print('Data size',len(words))
# print(words)

#创建词汇表，将出现最多的50000个单词作为词汇表，放入字典中。
vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    # c=collections.Counter(words).most_common(10)
    # print(c)
    # count.extend(c)
    # print(count)  #[['UNK', -1], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764), ('in', 372201), ('a', 325873), ('to', 316376), ('zero', 264975), ('nine', 250430), ('two', 192644)]
    dictionary = dict()#新建空字典
    for word,_ in count:
        dictionary[word] = len(dictionary)
    # print(dictionary)  #{'UNK': 0, 'the': 1, 'of': 2, 'and': 3, 'one': 4, 'in': 5, 'a': 6, 'to': 7, 'zero': 8, 'nine': 9, 'two': 10}
    data = list()
    unk_count = 0#未知单词数量
    for word in words:#单词索引，不在字典中，则索引为0
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reverse_dictionary

data,count,dictionary,reverse_dictionary = build_dataset(words)

#删除原始单词列表，节约内存。打印词汇表，了解词频
del words
# print('Most common words (+UNK)',count[:5])
# print('Sample data',data[:10],[reverse_dictionary[i] for i in data[:10]])

#以上代码为数据处理，得到单词的词频和在字典中的索引

#skip-gram模式：从目标单词反推语境

data_index = 0

#生成训练用的batch数据
#batch_size为batch大小，num_skips为对每个单词生成样本数，skip_window为单词最远可以联系的距离
def generate_batch(batch_size,num_skips,skip_window):
    global data_index  #声明全局变量
    assert batch_size % num_skips == 0#断言batch_size是num_skips的整倍数
    assert num_skips <= 2 * skip_window#断言num_skips不大于skip_window的两倍
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)#初始化为数组
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span = 2 * skip_window + 1  #对某个单词创建相关样本时会使用到的单词数量
    buffer = collections.deque(maxlen=span)  #创建最大容量为span的队列，即双向队列 collections

    for _ in range(span): # add span
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):#'//'取商的整数部分
        target = skip_window
        targets_to_avoid = [skip_window]#因为要预测语境单词，不包括目标单词本身。所以需要一个避免列表
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch,labels


# batch,labels = generate_batch(batch_size=8,num_skips=2,skip_window=1)
# print(batch)#[3081 3081   12   12    6    6  195  195]
# print(labels)#[[5234]
#              # [  12]
#              # [3081]
#              # [   6]
#              # [  12]
#              # [ 195]
#              # [   6]
#              # [   2]]
# for i in range(8):
#     print(batch[i],reverse_dictionary[batch[i]],'->',labels[i,0],reverse_dictionary[labels[i,0]])


batch_size = 128
embedding_size = 128#将单词转为稠密向量的维度，一般在50~1000范围
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window,valid_size,replace=False)#生成验证数据，随机抽取词频最高（前valid_window）的valid_size个单词
num_sampled = 64#做负样本的噪声单词数量

#定义skip-gram网络结构
graph = tf.Graph()
with graph.as_default():

    train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
    train_labels = tf.placeholder(tf.int32,shape=[batch_size,1])
    valid_dataset = tf.constant(valid_examples,dtype=tf.int32)

#限定所有计算都在cpu上执行，因为接下来一些计算操作在GPU上可能还没有实现
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))#随机生成所有单词的词向量，单词表大小50000，维度128
        embed = tf.nn.embedding_lookup(embeddings,train_inputs)#查找输入train_inputs在embeddings里对应的向量
        #用截断正态分布truncated_normal初始化NCE Loss中的权重参数nce_weights，并将其初始化为0
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,labels=train_labels,inputs=embed,num_sampled=num_sampled,num_classes=vocabulary_size))

    #优化器SGD，学习率1.0
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    #先计算embeddings的平方，并按第二维降维到1，计算嵌入向量embeddings的L2范数
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
    #标准化embeddings
    normalized_embeddings = embeddings/norm
    #查询单词的嵌入向量，并计算验证单词的嵌入向量与词汇表中所有单词的相似性
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
    #transpose_b=True  将b转置
    similarity = tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)
    #初始化所有模型参数
    init = tf.global_variables_initializer()

num_steps = 100001#迭代10万次

with tf.Session(graph=graph) as session:
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs,batch_labels = generate_batch(batch_size,num_skips,skip_window)
        feed_dict = {train_inputs : batch_inputs,train_labels : batch_labels}

        _,loss_val = session.run([optimizer,loss],feed_dict=feed_dict)

        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:

                average_loss /= 2000
            print('Average loss at step ',step,': ',average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k+1]#argsort将数组从小到大排列，并返回索引
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str,close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


from sklearn.manifold import TSNE#此降维算法比PCA更高级，可视化
import matplotlib.pyplot as plt

def plot_with_labels(low_dim_embs,labels,filename='tsne.png'):
    assert  low_dim_embs.shape[0] >= len(labels),'More labels than embeddings'
    plt.figure(figsize=(18,18))
    for i,label in enumerate(labels):#enumerate枚举可遍历、迭代（列表、字符串）对象，加上索引
        x,y = low_dim_embs[i,:]
        plt.scatter(x,y)#显示散点图
        #（工具书p242）annotate在图上添加注释，xy设置箭头所指处的坐标，xytext注释内容坐标，textcoords注释内容坐标的坐标变换方式。
        #'offset points'以点为单位，相对于点xy的坐标
        # ha='right'点在注释右边（right,center,left），va='bottom'点在注释底部('top', 'bottom', 'center', 'baseline')
        plt.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')

    plt.savefig(filename)

#perplexity（混乱，复杂）与最近邻数有关，一般在5~50，n_iter达到最优化所需的最大迭代次数，应当不少于250次
#init='pca'pca初始化比random稳定，n_components嵌入空间的维数（即降到2维，默认为2
tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
plot_only = 100#显示词频最高的一百个
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs,labels)

plt.show()
