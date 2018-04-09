# -*- coding: utf-8 -*-
"""
@author: duangan

循环神经网络
"""

import numpy as np
import tensorflow as tf
import reader

DATA_PATH = "/Volumes/Data/TensorFlow/datasets/PTB/data"
# 隐层规模，即隐层状态向量的长度。
HIDDEN_SIZE = 200
# LSTM结构的层数
NUM_LAYERS = 2
# 单词标记符，总共一万个单词。
VOCAB_SIZE = 10000

# 学习速率
LEARNING_RATE = 1.0
# batch 大小
TRAIN_BATCH_SIZE = 20
# 训练数据截断大小
TRAIN_NUM_STEP = 35

# 测试数据batch的大小
EVAL_BATCH_SIZE = 1
# 测试数据的截断长度。1表示不截断。在测试时不需要使用截断。
EVAL_NUM_STEP = 1
# 使用训练数据的轮数
NUM_EPOCH = 2
# 节点不被dropout的概率
KEEP_PROB = 0.5
# 用于控制梯度膨胀的参数
MAX_GRAD_NORM = 5


# 定义一个类来描述模型结构。目的是方便维护rnn中的状态。
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        
        self.batch_size = batch_size
        self.num_steps = num_steps
        
        # 定义输入层。
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        # 定义预期输出。
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        
        # 定义LSTM结构。
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)

        # 只对训练集使用 dropout。
        if is_training:
        	# 指定 dropout 参数
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)

        # 使用LSTM结构定义rnn神经网络。
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*NUM_LAYERS)
        
        # 初始化最初的状态为全零。
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        
        # 计算单词向量的维度（此次为200万）。总共有VOCAB_SIZE个单词，每个单词的向量维度为HIDDEN_SIZE。
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        # 将单词ID转为单词向量。转化后的输入层维度为 
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # 定义输出列表。先将不同时刻的LSTM结构的输出收集起来，再通过一个全连接层得到最终的输出。
        outputs = []
        # state负责存储不同batch中LSTM的状态。将其初始化为0.
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                # 从输入数据中获取当前时刻获取的数量，并传入LSTM结构。
                cell_output, state = cell(inputs[:, time_step, :], state)
                # 将当前输出加入输出队列。
                outputs.append(cell_output)

        # 将输出队列展开为[batch, hidden_size * num_steps]的形状，然后再
        # reshape 为[batch * num_steps, hidden_size]
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        # 将从LSTM中得到的输出再经过一个全连接层得到最后的预测结构，
        # 最终的预测结果在每一个时刻上都是一个长度为 VOCAB_SIZE 的数组，
        # 通过 softmax 层之后表示下一个位置是不同单词的概率。
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias
        
        # 定义交叉熵损失函数和平均损失。
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        	# 预测的结果
            [logits],
            # 预测的正确结果，这里将[batch, num_steps]二维数组压缩成一维数组。
            [tf.reshape(self.targets, [-1])],
            # 损失的权重，在这里所有的权重都为1，也就是说不同的batch和不同时刻的重要程序是一样的。
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])

        # 计算得到每个batch的平均损失
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state
        
        # 只在训练模型时定义反向传播操作。
        if not is_training: return
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小。通过 clip_by_global_norm() 控制梯度大小，避免梯度膨胀的问题。
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        # 定义优化方法。
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        # 定义训练步骤。
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


# 使用给定的模型model在数据data上运行train_op并返回在全部数据上的perplexity值
def run_epoch(session, model, data, train_op, output_log, epoch_size):
	# 计算 perplexity 的辅助变量
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 训练一个 epoch 。使用当前数据训练或测试模型。
    for step in range(epoch_size):
        x, y = session.run(data)
        # 在当前batch上运行 train_op 并计算损失值。交叉熵损失函数计算的就是下一个单词为给定单词的概率。
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                        {model.input_data: x, model.targets: y, 
                                        model.initial_state: state})
        # 将不同时刻、不同batch的概率加起来就可以得到第二个 perplexity 公式等号右边的部分，
        # 再将这个和指数运算就可以得到 perplexity 值。
        total_costs += cost
        iters += model.num_steps

        # 在训练时输出日志。
        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))
    # 返回给定模型在给定数据上的 perplexity 值。 
    return np.exp(total_costs / iters)


def main():
	# 获取原始数据
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    # 训练集
    train_data_len = len(train_data)
    train_batch_len = train_data_len // TRAIN_BATCH_SIZE
    train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP

    # 验证集
    valid_data_len = len(valid_data)
    valid_batch_len = valid_data_len // EVAL_BATCH_SIZE
    valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP

    # 测试集
    test_data_len = len(test_data)
    test_batch_len = test_data_len // EVAL_BATCH_SIZE
    test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP

    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的RNN
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    # 定义验证用的RNN
    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # 训练模型开始
    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_queue = reader.ptb_producer(train_data, train_model.batch_size, train_model.num_steps)
        eval_queue = reader.ptb_producer(valid_data, eval_model.batch_size, eval_model.num_steps)
        test_queue = reader.ptb_producer(test_data, eval_model.batch_size, eval_model.num_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        # 训练轮次
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            # 在所有训练数据上训练
            run_epoch(session, train_model, train_queue, train_model.train_op, True, train_epoch_size)

            # 使用验证数据评测效模型效果。
            valid_perplexity = run_epoch(session, eval_model, eval_queue, tf.no_op(), False, valid_epoch_size)
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))

        # 使用测试数据测试模型效果。
        test_perplexity = run_epoch(session, eval_model, test_queue, tf.no_op(), False, test_epoch_size)
        print("Test Perplexity: %.3f" % test_perplexity)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()





