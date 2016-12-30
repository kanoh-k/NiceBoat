import racer
import raceresults
import tensorflow as tf
import numpy as np
import os

class NiceBoat:
    def __init__(self):
        # レース情報
        self.race_results = raceresults.RaceResults()
        self.race_results.load()

        # ネットワークのパラメータ
        self.input_layer_size = self.race_results.get_input_length() * 6 # 1672 * 6 for the last period
        self.hidden_layer_sizes = [1000]
        self.output_layer_size = 18
        self.batch_size = 128
        self.epochs = 30
        self.learning_rate = 0.001

    def prepare_data(self, train_ratio=0.8):
        _X, Y, odds = zip(*self.race_results.results)

        X = []
        for x in _X:
            X.append([self.race_results.get_input(n) for n in x])

        # Make batch with self.batch_size races
        batchedX = [X[x:x+self.batch_size] for x in range(0, len(X), self.batch_size)]
        batchedX = np.array([x for x in batchedX if len(x) == self.batch_size])
        batchedY = [Y[x:x+self.batch_size] for x in range(0, len(Y), self.batch_size)]
        batchedY = np.array([x for x in batchedY if len(x) == self.batch_size])
        batchedOdds = [odds[x:x+self.batch_size] for x in range(0, len(odds), self.batch_size)]
        batchedOdds = np.array([x for x in batchedOdds if len(x) == self.batch_size])

        # Divide data into training and test
        train_num = int(len(batchedX) * train_ratio)
        return batchedX[:train_num], batchedY[:train_num], batchedOdds[:train_num], batchedX[train_num:], batchedY[train_num:], batchedOdds[train_num:]

    def convert_input(self, input_data):
        onehots = tf.map_fn(lambda x: tf.one_hot(x, self.race_results.get_input_length(), dtype=tf.int32), input_data)
        return tf.cast(tf.reshape(onehots, [-1, self.race_results.get_input_length() * 6]), "float")

    def convert_label(self, label):
        onehots = tf.map_fn(lambda x: tf.one_hot(x, 6, dtype=tf.int32), label)
        return tf.cast(tf.reshape(onehots, [-1, 6 * 3]), "float")

    def inference(self, input_data):
        layers = [self.input_layer_size] + self.hidden_layer_sizes
        previous_layer = input_data
        for i in range(len(layers) - 1):
            ws = tf.Variable(tf.truncated_normal([layers[i], layers[i + 1]], stddev=0.001))
            bs = tf.Variable(tf.ones([layers[i + 1]]))
            previous_layer = tf.nn.relu(tf.matmul(previous_layer, ws) + bs)

        wo = tf.Variable(tf.truncated_normal([layers[-1], self.output_layer_size], stddev=0.001))
        bo = tf.Variable(tf.ones([self.output_layer_size]))
        output = tf.matmul(previous_layer, wo) + bo
        return output

    def loss(self, output, actual_labels):
        # output: (batch_size, output_layer_size)
        osize = self.output_layer_size // 3
        first = tf.slice(output, [0, 0], [self.batch_size, osize])
        second =tf.slice(output, [0, osize], [self.batch_size, osize])
        third = tf.slice(output, [0, osize * 2], [self.batch_size, osize])
        label1 = tf.slice(actual_labels, [0, 0], [self.batch_size, osize])
        label2 = tf.slice(actual_labels, [0, osize], [self.batch_size, osize])
        label3 = tf.slice(actual_labels, [0, osize * 2], [self.batch_size, osize])
        cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(first, label1))
        cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(second, label2))
        cost3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(third, label3))
        return tf.add_n([cost1, cost2, cost3])

    def training(self, cost):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return optimizer

    def softmax(self, output):
        osize = self.output_layer_size // 3
        first = tf.nn.softmax(tf.slice(output, [0, 0], [-1, osize]))
        second =tf.nn.softmax(tf.slice(output, [0, osize], [-1, osize]))
        third = tf.nn.softmax(tf.slice(output, [0, osize * 2], [-1, osize]))
        return tf.concat_v2([first, second, third], 1)

    # 単勝
    def accuracy_win(self, output, actual_labels):
        osize = self.output_layer_size // 3
        first = tf.slice(output, [0, 0], [self.batch_size, osize])
        label1 = tf.slice(actual_labels, [0, 0], [self.batch_size, osize])

        correct = tf.equal(tf.argmax(first, 1), tf.argmax(label1, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        return accuracy

    # 複勝
    def accuracy_place(self, output, actual_labels):
        osize = self.output_layer_size // 3
        first = tf.slice(output, [0, 0], [self.batch_size, osize])
        label1 = tf.slice(actual_labels, [0, 0], [self.batch_size, osize])
        label2 = tf.slice(actual_labels, [0, osize], [self.batch_size, osize])

        correct = tf.logical_or(tf.equal(tf.argmax(first, 1), tf.argmax(label1, 1)), tf.equal(tf.argmax(first, 1), tf.argmax(label2, 1)))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        return accuracy

    # 2連単
    def accuracy_exacta(self, output, actual_labels):
        osize = self.output_layer_size // 3
        first = tf.slice(output, [0, 0], [self.batch_size, osize])
        second =tf.slice(output, [0, osize], [self.batch_size, osize])
        label1 = tf.slice(actual_labels, [0, 0], [self.batch_size, osize])
        label2 = tf.slice(actual_labels, [0, osize], [self.batch_size, osize])

        correct = tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label1, 1)), tf.equal(tf.argmax(second, 1), tf.argmax(label2, 1)))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        return accuracy

    # 2連複
    def accuracy_quinella(self, output, actual_labels):
        osize = self.output_layer_size // 3
        first = tf.slice(output, [0, 0], [self.batch_size, osize])
        second =tf.slice(output, [0, osize], [self.batch_size, osize])
        label1 = tf.slice(actual_labels, [0, 0], [self.batch_size, osize])
        label2 = tf.slice(actual_labels, [0, osize], [self.batch_size, osize])

        correct = tf.logical_or(
            tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label1, 1)), tf.equal(tf.argmax(second, 1), tf.argmax(label2, 1))),
            tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label2, 1)), tf.equal(tf.argmax(second, 1), tf.argmax(label1, 1))),
            )
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        return accuracy

    # 3連単
    def accuracy_trifecta(self, output, actual_labels):
        osize = self.output_layer_size // 3
        first = tf.slice(output, [0, 0], [self.batch_size, osize])
        second =tf.slice(output, [0, osize], [self.batch_size, osize])
        third = tf.slice(output, [0, osize * 2], [self.batch_size, osize])
        label1 = tf.slice(actual_labels, [0, 0], [self.batch_size, osize])
        label2 = tf.slice(actual_labels, [0, osize], [self.batch_size, osize])
        label3 = tf.slice(actual_labels, [0, osize * 2], [self.batch_size, osize])

        correct = tf.logical_and(
            tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label1, 1)), tf.equal(tf.argmax(second, 1), tf.argmax(label2, 1))),
            tf.equal(tf.argmax(third, 1), tf.argmax(label3, 1))
            )
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        return accuracy

    # 3連複
    def accuracy_trio(self, output, actual_labels):
        osize = self.output_layer_size // 3
        first = tf.slice(output, [0, 0], [self.batch_size, osize])
        second =tf.slice(output, [0, osize], [self.batch_size, osize])
        third = tf.slice(output, [0, osize * 2], [self.batch_size, osize])
        label1 = tf.slice(actual_labels, [0, 0], [self.batch_size, osize])
        label2 = tf.slice(actual_labels, [0, osize], [self.batch_size, osize])
        label3 = tf.slice(actual_labels, [0, osize * 2], [self.batch_size, osize])

        correct = tf.logical_or(
            tf.logical_or(
                tf.logical_or(
                    tf.logical_and(
                        tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label1, 1)), tf.equal(tf.argmax(second, 1), tf.argmax(label2, 1))),
                        tf.equal(tf.argmax(third, 1), tf.argmax(label3, 1))),
                    tf.logical_and(
                        tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label1, 1)), tf.equal(tf.argmax(second, 1), tf.argmax(label3, 1))),
                        tf.equal(tf.argmax(third, 1), tf.argmax(label2, 1)))),
                tf.logical_or(
                    tf.logical_and(
                        tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label2, 1)), tf.equal(tf.argmax(second, 1), tf.argmax(label1, 1))),
                        tf.equal(tf.argmax(third, 1), tf.argmax(label3, 1))),
                    tf.logical_and(
                        tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label2, 1)), tf.equal(tf.argmax(second, 1), tf.argmax(label3, 1))),
                        tf.equal(tf.argmax(third, 1), tf.argmax(label1, 1))))),
            tf.logical_or(
                    tf.logical_and(
                        tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label3, 1)), tf.equal(tf.argmax(second, 1), tf.argmax(label1, 1))),
                        tf.equal(tf.argmax(third, 1), tf.argmax(label2, 1))),
                    tf.logical_and(
                        tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label3, 1)), tf.equal(tf.argmax(second, 1), tf.argmax(label2, 1))),
                        tf.equal(tf.argmax(third, 1), tf.argmax(label1, 1)))))

        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        return accuracy

    # 拡連複
    def accuracy_wide_quinella(self, output, actual_labels):
        osize = self.output_layer_size // 3
        first = tf.slice(output, [0, 0], [self.batch_size, osize])
        second =tf.slice(output, [0, osize], [self.batch_size, osize])
        third = tf.slice(output, [0, osize * 2], [self.batch_size, osize])
        label1 = tf.slice(actual_labels, [0, 0], [self.batch_size, osize])
        label2 = tf.slice(actual_labels, [0, osize], [self.batch_size, osize])

        correct = tf.logical_or(
            tf.logical_or(
                tf.logical_or(
                    tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label1, 1)), tf.equal(tf.argmax(second, 1), tf.argmax(label2, 1))),
                    tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label2, 1)), tf.equal(tf.argmax(second, 1), tf.argmax(label1, 1)))),
                tf.logical_or(
                    tf.logical_and(tf.equal(tf.argmax(second, 1), tf.argmax(label1, 1)), tf.equal(tf.argmax(third, 1), tf.argmax(label2, 1))),
                    tf.logical_and(tf.equal(tf.argmax(second, 1), tf.argmax(label2, 1)), tf.equal(tf.argmax(third, 1), tf.argmax(label1, 1))))),
            tf.logical_or(
                tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label1, 1)), tf.equal(tf.argmax(third, 1), tf.argmax(label2, 1))),
                tf.logical_and(tf.equal(tf.argmax(first, 1), tf.argmax(label2, 1)), tf.equal(tf.argmax(third, 1), tf.argmax(label1, 1)))))

        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        return accuracy

    def train(self):
        # input_data = self.convert_input(tf.placeholder("int32", [None, 6]))
        # actual_labels = self.convert_label(tf.placeholder("int32", [None, 3]))
        input_data = tf.placeholder("int32", [None, 6])
        actual_labels = tf.placeholder("int32", [None, 3])

        one_hot_inputs = self.convert_input(input_data)
        one_hot_labels = self.convert_label(actual_labels)
        print("One hot input:", one_hot_inputs)

        prediction = self.inference(one_hot_inputs)
        cost = self.loss(prediction, one_hot_labels)
        optimizer = self.training(cost)

        tf.summary.scalar("Cross entropy", cost)
        summary = tf.summary.merge_all()
        dirname = "-".join([str(x) for x in self.hidden_layer_sizes]) + "_" + str(self.learning_rate)

        print("Preparing training data...")
        trainX, trainY, _, testX, testY, _ = self.prepare_data()
        print("#train:", trainX.shape[0] * trainX.shape[1])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter("./log/" + dirname, sess.graph)

            for epoch in range(self.epochs):
                step = 0
                epoch_loss = 0

                for batchX, batchY in zip(trainX, trainY):
                    _, c = sess.run([optimizer, cost], feed_dict={input_data: batchX, actual_labels: batchY})
                    epoch_loss += c
                    step += 1

                print("Epoch", epoch, "completed out of", self.epochs, "-- loss:", epoch_loss)

                summary_str = sess.run(summary, feed_dict={input_data: trainX.reshape([trainX.shape[0] * trainX.shape[1], trainX.shape[2]]), actual_labels: trainY.reshape([trainY.shape[0] * trainY.shape[1], trainY.shape[2]])})
                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()

            saver = tf.train.Saver()
            saver.save(sess, "./model/model.ckpt")

    def evaluate(self, modelfile="./model/model.ckpt"):
        input_data = tf.placeholder("int32", [None, 6])
        actual_labels = tf.placeholder("int32", [None, 3])
        one_hot_inputs = self.convert_input(input_data)
        one_hot_labels = self.convert_label(actual_labels)

        prediction = self.inference(one_hot_inputs)

        accuracy_win = self.accuracy_win(prediction, one_hot_labels)
        accuracy_place = self.accuracy_place(prediction, one_hot_labels)
        accuracy_exacta = self.accuracy_exacta(prediction, one_hot_labels)
        accuracy_quinella = self.accuracy_quinella(prediction, one_hot_labels)
        accuracy_trifecta = self.accuracy_trifecta(prediction, one_hot_labels)
        accuracy_trio = self.accuracy_trio(prediction, one_hot_labels)
        accuracy_wide_quinella = self.accuracy_wide_quinella(prediction, one_hot_labels)

        print("Preparing train/test data...")
        trainX, trainY, _, testX, testY, _ = self.prepare_data()
        trainX = trainX.reshape([trainX.shape[0] * trainX.shape[1], trainX.shape[2]])
        trainY = trainY.reshape([trainY.shape[0] * trainY.shape[1], trainY.shape[2]])
        testX = testX.reshape([testX.shape[0] * testX.shape[1], testX.shape[2]])
        testY = testY.reshape([testY.shape[0] * testY.shape[1], testY.shape[2]])
        train_dict = {input_data: trainX, actual_labels: trainY}
        test_dict = {input_data: testX, actual_labels: testY}
        print("#train:", len(trainX), "#test:", len(testX))

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, modelfile)

            win_train, place_train, exacta_train, quinella_train, trifecta_train, trio_train, wide_quinella_train = sess.run(
                [accuracy_win, accuracy_place, accuracy_exacta, accuracy_quinella, accuracy_trifecta, accuracy_trio, accuracy_wide_quinella],
                feed_dict=train_dict)
            win_test, place_test, exacta_test, quinella_test, trifecta_test, trio_test, wide_quinella_test = sess.run(
                [accuracy_win, accuracy_place, accuracy_exacta, accuracy_quinella, accuracy_trifecta, accuracy_trio, accuracy_wide_quinella],
                feed_dict=test_dict)
            print("*** Evaluation on train ***")
            print("単勝:", win_train)
            print("複勝:", place_train)
            print("2連単:", exacta_train)
            print("2連複:", quinella_train)
            print("3連単:", trifecta_train)
            print("3連複:", trio_train)
            print("拡連複:", wide_quinella_train)
            print("\n*** Evaluation on test ***")
            print("単勝:", win_test)
            print("複勝:", place_test)
            print("2連単:", exacta_test)
            print("2連複:", quinella_test)
            print("3連単:", trifecta_test)
            print("3連複:", trio_test)
            print("拡連複:", wide_quinella_test)

    def output2prediction(self, output):
        results = np.zeros([output.shape[0], 3], dtype=int)
        for idx in range(output.shape[0]):
            vec = output[idx]
            used = []
            for i in range(3):
                m = 0
                argmax = 0
                for j in range(6):
                    if j in used: continue
                    if m < vec[i * 6 + j]:
                        m = vec[i * 6 + j]
                        argmax = j
                results[idx][i] = argmax
                used.append(argmax)
        return results

    def backtest(self, modelfile="./model/model.ckpt"):
        input_data = tf.placeholder("int32", [None, 6])
        actual_labels = tf.placeholder("int32", [None, 3])
        one_hot_inputs = self.convert_input(input_data)
        one_hot_labels = self.convert_label(actual_labels)

        prediction = self.softmax(self.inference(one_hot_inputs))

        print("Preparing train/test data...")
        trainX, trainY, trainOdds, testX, testY, testOdds = self.prepare_data()
        trainX = trainX.reshape([trainX.shape[0] * trainX.shape[1], trainX.shape[2]])
        trainY = trainY.reshape([trainY.shape[0] * trainY.shape[1], trainY.shape[2]])
        trainOdds = trainOdds.reshape([trainOdds.shape[0] * trainOdds.shape[1], trainOdds.shape[2], trainOdds.shape[3]])
        testX = testX.reshape([testX.shape[0] * testX.shape[1], testX.shape[2]])
        testY = testY.reshape([testY.shape[0] * testY.shape[1], testY.shape[2]])
        testOdds = testOdds.reshape([testOdds.shape[0] * testOdds.shape[1], testOdds.shape[2], testOdds.shape[3]])
        train_dict = {input_data: trainX, actual_labels: trainY}
        test_dict = {input_data: testX, actual_labels: testY}
        print("#train:", trainX.shape, "#test:", testX.shape)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, modelfile)

            output = sess.run(prediction, feed_dict=test_dict)
            prediction_label = self.output2prediction(output)

        with open("./backtest.csv", "w") as f:
            f.write("予想順位,実際の順位,単勝,複勝,2連単,2連複,3連単,3連複,拡連複,流し,ボックス\n")

            win_total = 0
            place_total = 0
            exacta_total = 0
            quinella_total = 0
            wide_quinella_total = 0
            trifecta_total = 0
            trio_total = 0
            for p, a, o in zip(prediction_label, testY, testOdds):
                p = [str(x + 1) for x in p]
                single = str(p[0])
                double = '-'.join(p[:2])
                double_sorted = '-'.join(sorted(p[:2]))
                triple = '-'.join(p)
                triple_sorted = '-'.join(sorted(p))

                # 単勝
                win = -100
                if single == o[0][0]: win += int(o[0][1])
                win_total += win

                # 複勝
                place = -100
                if single == o[1][0]: place += int(o[1][1])
                if single == o[2][0]: place += int(o[2][1])
                place_total += place

                # 2連単
                exacta = -100
                if double == o[3][0]: exacta += int(o[3][1])
                exacta_total += exacta

                # 2連複
                quinella = int(o[4][1]) if double_sorted == o[4][0] else -100
                quinella_total += quinella

                # 拡連複
                wide_quinella = -100
                if double_sorted == o[5][0]: wide_quinella += int(o[5][1])
                if double_sorted == o[6][0]: wide_quinella += int(o[6][1])
                if double_sorted == o[7][0]: wide_quinella += int(o[7][1])
                wide_quinella_total += wide_quinella

                # 3連単
                trifecta = -100
                if triple == o[8][0]: trifecta += int(o[8][1])
                trifecta_total += trifecta

                # 3連複
                trio = -100
                if triple_sorted == o[9][0]: trio += int(o[9][1])
                trio_total += trio


                f.write("%s,%s,%d,%d,%d,%d,%d,%d,%d\n" % (triple, '-'.join([str(x + 1) for x in a]), win, place, exacta, quinella, wide_quinella, trifecta, trio))
            print("通算利益")
            print("単勝:", win_total)
            print("複勝:", place_total)
            print("2連単:", exacta_total)
            print("2連複:", quinella_total)
            print("拡連複:", wide_quinella_total)
            print("3連単:", trifecta_total)
            print("3連複:", trio_total)

    def predict(self, inputs, modelfile="./model/model.ckpt"):
        input_data = tf.placeholder("int32", [None, 6])
        one_hot_inputs = self.convert_input(input_data)
        prediction = self.softmax(self.inference(one_hot_inputs))

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, modelfile)

            output = sess.run(prediction, feed_dict={input_data: inputs})
            prediction_label = self.output2prediction(output)

        for label in prediction_label:
            print('-'.join([str(x + 1) for x in label]))

nb = NiceBoat()
# nb.train()
# nb.evaluate()
# nb.backtest()
# nb.predict([[3674,3611,4450,4243,4534,3894]])
