
import tensorflow as tf
import dataset as dataset


LEARNING_RATE = 1e-4
BATCH_SIZE = 200
EPOCH = 50
EPOCH_PER_EVAL = 10
DATA_DIR = 'data'
MODEL_DIR = 'model'


def build_graph(inputs, is_train=True):
    with tf.variable_scope("main_graph", initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE):
        inputs = tf.reshape(inputs, [BATCH_SIZE, 28, 28, 1])
        inputs = tf.layers.conv2d(
            inputs,
            32,
            [5,5],
            padding='same',
            activation=tf.nn.relu
        )
        inputs = tf.layers.max_pooling2d(
            inputs,
            (2, 2),
            (2, 2),
            padding='same'
        )
        inputs = tf.layers.conv2d(
            inputs,
            64,
            [5,5],
            padding='same',
            activation=tf.nn.relu
        )
        inputs = tf.layers.max_pooling2d(
            inputs,
            (2, 2),
            (2, 2),
            padding='same'
        )
        inputs = tf.reshape(inputs, [BATCH_SIZE, -1])
        inputs = tf.layers.dense(inputs, 1024, activation=tf.nn.relu)
        if is_train:
            inputs = tf.layers.dropout(inputs, rate=0.4)
        logits = tf.layers.dense(inputs, 10, activation=tf.nn.relu)
    return logits


def model_fn(features, labels, mode, params):
    image = features
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = build_graph(image, is_train=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        logits = build_graph(image, is_train=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1))
        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(LEARNING_RATE, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        # Save accuracy scalar to Tensorboard output.
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
    if mode == tf.estimator.ModeKeys.EVAL:
        logits = build_graph(image, is_train=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy':
                    tf.metrics.accuracy(
                        labels=labels, predictions=tf.argmax(logits, axis=1)),
            })


def main(_):
    model_function = model_fn
    session_config = tf.ConfigProto(allow_soft_placement=True)
    run_config = tf.estimator.RunConfig(
        session_config=session_config,
        keep_checkpoint_max=EPOCH
    )
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=MODEL_DIR,
        config=run_config
    )

    # Train and evaluate model.
    for _ in range(EPOCH // EPOCH_PER_EVAL):
        mnist_classifier.train(input_fn=get_input_fn('train'))
        eval_results = mnist_classifier.evaluate(input_fn=get_input_fn('valid'))
        print('\nEvaluation results:\n\t%s\n' % eval_results)


def get_input_fn(mode="train"):
    assert mode in ["train", "valid", "test"]
    if mode == "train":
        def input_fn():

            ds = dataset.train(DATA_DIR)
            ds = ds.cache().shuffle(buffer_size=50000).batch(BATCH_SIZE)
            ds = ds.repeat(EPOCH_PER_EVAL)
            return ds
    elif mode == "valid":

        def input_fn():
            return dataset.test(DATA_DIR).batch(BATCH_SIZE)
    else:
        def input_fn():
            return dataset.test(DATA_DIR).batch(BATCH_SIZE)
    return input_fn


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)





